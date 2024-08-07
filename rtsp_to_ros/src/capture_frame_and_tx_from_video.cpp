#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <gst/gst.h>
#include <glib.h>
#include <gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sys/time.h>
#include <stdint.h>
#include <atomic>
#include <mutex>

#define MAX_ACCUMULATED_SIZE 1048576 // 1MB max accumulation buffer

static GMainLoop *loop = NULL;
static GstElement *pipeline = NULL;
static ros::Publisher compressed_image_pub;
static double publish_frequency;
static ros::Time last_publish_time;
static guint8 *accumulated_buffer = NULL;
static gsize accumulated_size = 0;
static const char* WINDOW_NAME = "RTSP Video Feed";
static bool resize_display = true;

// Shared timestamp variable
static std::atomic<uint64_t> latest_timestamp_ms(0);
static std::mutex timestamp_mutex;

static void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
    GstElement *sink = (GstElement *)data;
    GstPad *sinkpad = gst_element_get_static_pad(sink, "sink");
    
    if (!gst_pad_is_linked(sinkpad)) {
        if (gst_pad_link(pad, sinkpad) != GST_PAD_LINK_OK) {
            ROS_ERROR("Failed to link pads");
        } else {
            ROS_INFO("Pads linked successfully");
        }
    }
    
    gst_object_unref(sinkpad);
}

static GstPadProbeReturn h264_parse_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    GstBuffer *buffer;
    GstMapInfo map;
    guint8 *data;
    gsize size, total_size;
    gint i = 0;

    buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        ROS_ERROR("Failed to map buffer");
        return GST_PAD_PROBE_OK;
    }

    data = map.data;
    size = map.size;

    ROS_DEBUG("Received buffer of size: %zu", size);

    // Accumulate buffer
    total_size = accumulated_size + size;
    if (total_size > MAX_ACCUMULATED_SIZE) {
        ROS_WARN("Accumulated buffer overflow, resetting");
        accumulated_size = 0;
    }

    if (accumulated_buffer == NULL) {
        accumulated_buffer = (guint8*)g_malloc(MAX_ACCUMULATED_SIZE);
        if (accumulated_buffer == NULL) {
            ROS_ERROR("Failed to allocate accumulated buffer");
            gst_buffer_unmap(buffer, &map);
            return GST_PAD_PROBE_OK;
        }
    }

    memcpy(accumulated_buffer + accumulated_size, data, size);
    accumulated_size = total_size;

    // Process accumulated buffer
    guint8 *process_data = accumulated_buffer;
    gsize process_size = accumulated_size;

    while (i + 4 <= process_size) {
        guint32 nal_size = (process_data[i] << 24) | (process_data[i+1] << 16) | 
                           (process_data[i+2] << 8) | process_data[i+3];
        
        if (i + 4 + nal_size <= process_size) {
            guint8 nal_type = process_data[i+4] & 0x1F;

            if (nal_type == 6) {  // SEI NAL unit
                // Check for our specific SEI payload type
                if (i + 6 < process_size && process_data[i+5] == 0x05) {
                    // Try to extract timestamp
                    if (i + 15 <= process_size) {  // 4 bytes length + 1 byte NAL header + 1 byte payload type + 1 byte payload size + 8 bytes timestamp
                        uint64_t timestamp_ms;
                        memcpy(&timestamp_ms, &process_data[i+7], sizeof(uint64_t));

                        // Update the shared timestamp
                        latest_timestamp_ms.store(timestamp_ms);
                        
                        struct timeval tv;
                        gettimeofday(&tv, NULL);
                        uint64_t current_time_ms = (uint64_t)(tv.tv_sec) * 1000 + (uint64_t)(tv.tv_usec) / 1000;
                        ROS_INFO("Timestamp & delay (ms): %lu, Difference: %ld ms", 
                                timestamp_ms, (long)(current_time_ms - timestamp_ms));
                    }
                }
            }

            i += 4 + nal_size;
        } else {
            // Incomplete NAL unit, keep it in the buffer
            break;
        }
    }

    // Remove processed data from accumulated buffer
    if (i > 0) {
        memmove(accumulated_buffer, accumulated_buffer + i, accumulated_size - i);
        accumulated_size -= i;
    }

    gst_buffer_unmap(buffer, &map);
    return GST_PAD_PROBE_OK;
}

static GstFlowReturn new_sample(GstElement *sink, gpointer user_data) {
    GstSample *sample;
    GstBuffer *buffer;
    GstMapInfo map;
    cv::Mat frame, display_frame;

    g_signal_emit_by_name(sink, "pull-sample", &sample);
    if (sample) {
        buffer = gst_sample_get_buffer(sample);
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            GstCaps *caps = gst_sample_get_caps(sample);
            GstStructure *structure = gst_caps_get_structure(caps, 0);
            int width, height;
            const gchar *format;
            gst_structure_get_int(structure, "width", &width);
            gst_structure_get_int(structure, "height", &height);
            format = gst_structure_get_string(structure, "format");

            // ROS_INFO("Received frame with dimensions: %dx%d, format: %s", width, height, format);

            if (strcmp(format, "YUY2") == 0) {
                // YUY2 format: YUYV 4:2:2 packed
                cv::Mat yuyv_frame(height, width, CV_8UC2, (char*)map.data);
                cv::cvtColor(yuyv_frame, frame, cv::COLOR_YUV2BGR_YUY2);
            } else if (strcmp(format, "I420") == 0) {
                // I420 format: YUV 4:2:0 planar
                cv::Mat yuv_frame(height * 3 / 2, width, CV_8UC1, (char*)map.data);
                cv::cvtColor(yuv_frame, frame, cv::COLOR_YUV2BGR_I420);
            } else {
                ROS_ERROR("Unsupported video format: %s", format);
                gst_buffer_unmap(buffer, &map);
                gst_sample_unref(sample);
                return GST_FLOW_ERROR;
            }

            // if (frame.empty()) {
            //     ROS_ERROR("Failed to create frame");
            //     gst_buffer_unmap(buffer, &map);
            //     gst_sample_unref(sample);
            //     return GST_FLOW_ERROR;
            // }

            // if (resize_display) {
            //     cv::resize(frame, display_frame, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
            // } else {
            //     display_frame = frame;
            // }

            // cv::imshow(WINDOW_NAME, display_frame);
            // int key = cv::waitKey(1);
            // if ((key == 'q' || key == 27) && loop != NULL) {
            //     g_main_loop_quit(loop);
            // }
            // ROS_INFO("Hello World");

            ros::Time current_time = ros::Time::now();
            if ((current_time - last_publish_time).toSec() >= (1.0 / publish_frequency)) {
                sensor_msgs::CompressedImagePtr compressed_msg = boost::make_shared<sensor_msgs::CompressedImage>();
                compressed_msg->header.stamp = ros::Time::now();
                compressed_msg->format = "jpeg";
                
                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
                compression_params.push_back(50); // 50% quality

                cv::imencode(".jpg", frame, compressed_msg->data, compression_params);

                uint64_t timestamp_ms = latest_timestamp_ms.load();
                compressed_msg->header.stamp.sec = timestamp_ms / 1000;
                compressed_msg->header.stamp.nsec = (timestamp_ms % 1000) * 1000000;

                // ROS_DEBUG("Publishing frame with timestamp: %lu ms", timestamp_ms);

                compressed_image_pub.publish(compressed_msg);
                last_publish_time = current_time;
                // ROS_INFO("Published compressed frame with timestamp: %f", compressed_msg->header.stamp.toSec());
            }

            gst_buffer_unmap(buffer, &map);
        }
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    return GST_FLOW_ERROR;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            ROS_INFO("End of stream");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            ROS_ERROR("Error: %s", error->message);
            g_error_free(error);
            g_free(debug);
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_STATE_CHANGED: {
            GstState old_state, new_state, pending_state;
            gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
            ROS_INFO("Pipeline state changed from %s to %s",
                     gst_element_state_get_name(old_state),
                     gst_element_state_get_name(new_state));
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "rtsp_to_ros_node");
    ros::NodeHandle nh("~");
    
    std::string rtsp_url;
    nh.param<std::string>("rtsp_url", rtsp_url, "rtsp://201.7.90.147:8554/test");
    nh.param<double>("publish_frequency", publish_frequency, 10.0);

    ROS_INFO("RTSP URL: %s", rtsp_url.c_str());
    ROS_INFO("Publish frequency: %f Hz", publish_frequency);

    compressed_image_pub = nh.advertise<sensor_msgs::CompressedImage>("video_frames/compressed", 10);
    last_publish_time = ros::Time::now();

    gst_init(&argc, &argv);

    // cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

    GstElement *source, *depay, *parse, *decoder, *convert, *sink;
    GstBus *bus;
    guint bus_watch_id;

    pipeline = gst_pipeline_new("rtsp-to-ros-pipeline");
    source = gst_element_factory_make("rtspsrc", "source");
    depay = gst_element_factory_make("rtph264depay", "depay");
    parse = gst_element_factory_make("h264parse", "parse");
    decoder = gst_element_factory_make("avdec_h264", "decoder");
    convert = gst_element_factory_make("videoconvert", "convert");
    sink = gst_element_factory_make("appsink", "sink");

    if (!pipeline || !source || !depay || !parse || !decoder || !convert || !sink) {
        ROS_ERROR("One or more elements could not be created. Exiting.");
        return -1;
    }

    g_object_set(G_OBJECT(source), "location", rtsp_url.c_str(), "latency", 0, NULL);
    g_object_set(G_OBJECT(sink), "emit-signals", TRUE, "sync", FALSE, NULL);

    gst_bin_add_many(GST_BIN(pipeline), source, depay, parse, decoder, convert, sink, NULL);
    if (!gst_element_link_many(depay, parse, decoder, convert, sink, NULL)) {
        ROS_ERROR("Elements could not be linked. Exiting.");
        gst_object_unref(pipeline);
        return -1;
    }

    g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), depay);
    g_signal_connect(sink, "new-sample", G_CALLBACK(new_sample), NULL);

    // Add probe to the src pad of rtph264depay
    GstPad *depay_src_pad = gst_element_get_static_pad(depay, "src");
    gst_pad_add_probe(depay_src_pad, GST_PAD_PROBE_TYPE_BUFFER, 
                      (GstPadProbeCallback)h264_parse_probe, NULL, NULL);
    gst_object_unref(depay_src_pad);

    bus = gst_element_get_bus(pipeline);
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    ROS_INFO("Setting pipeline to PLAYING state...");
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        ROS_ERROR("Failed to start up pipeline!");
        return -1;
    }

    ROS_INFO("Starting ROS spin...");
    ros::AsyncSpinner spinner(1);
    spinner.start();

    loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(loop);

    ROS_INFO("Stopping pipeline...");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    if (accumulated_buffer) {
        g_free(accumulated_buffer);
        accumulated_buffer = NULL;
    }

    cv::destroyAllWindows();

    return 0;
}