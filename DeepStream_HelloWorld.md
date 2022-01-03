# DeepStream

deepstream-test DeepStream 的 hello world

![](https://pic1.zhimg.com/80/v2-19572211c2bc6e891e55935eeaae6b84_720w.jpg)

## test1

### pipeline

首先数据源元件负责从磁盘上**读取**视频数据，解析器元件负责对数据进行**解析**，编码器元件负责对数据进行**解码**，流多路复用器元件负责**批处理**帧以实现最佳推理性能，推理元件负责实现**加速推理**，转换器元件负责将数据格式**转换**为输出显示支持的格式，可视化元件负责将边框与文本等信息**绘制**到图像中，渲染元件和接收器元件负责**输出**到屏幕上。  
![](https://pic1.zhimg.com/80/v2-f178b1cef853a6d19a2d2b079bb0f680_720w.jpg)

### code

#### main 函数

首先定义了需要用到的所有变量。因为所有 GStreamer 元件都具有相同的基类**GstElement**，因此能够采用 GstElement 类型对所有的元件进行定义。以及定义了负责数据消息的传输的 GstBus 类别变量。

```python
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
      *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
      *nvosd = NULL;
#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
```

然后在主函数中调用 gst_init()来完成相应的初始化工作，以便将用户从命令行输入的参数传递到 GStreamer 函数库。

```
gst_init (&argc, &argv);
```

创建主循环，在执行 g_main_loop_run 后正式开始循环

```
loop = g_main_loop_new (NULL, FALSE);
```

在 GStreamer 框架中，管道是用来容纳和管理元件的，下面创建一条名为 pipeline 的管道：

```
pipeline = gst_pipeline_new("dstest1-pipeline");
```

创建管理中需要使用的所有元件，最后检查所有元件是否创建成功

```
 // 创建一个gstreamer element, 类型为filesrc，名称为file-source。
  source = gst_element_factory_make ("filesrc", "file-source");

  // 创建一个gstreamer element, 类型为h264parse，名称为h264-parser。
  // 因为输入文件中的数据格式是基本的h264流，所以我们需要一个h264解析器
  h264parser = gst_element_factory_make ("h264parse", "h264-parser");

  // 创建一个gstreamer element, 类型为nvv4l2decoder，名称为nvv4l2-decoder。
  // 调用GPU硬件加速来解码h264文件
  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  // 创建一个gstreamer element, 类型为nvstreammux，名称为stream-muxer。
  // 从一个或多个源中来组成batches
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  // 若管道元件为空或者流复用器元件为空， 报错
  if (!pipeline || !streammux) {
  	g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  // 创建一个gstreamer element, 类型为nvinfer，名称为primary-nvinference-engine。
  // 使用nvinfer在解码器的输出上运行推理，推理过程的参数是通过配置文件设置的
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  // 创建一个gstreamer element, 类型为nvvideoconvert，名称为nvvideo-converter。
  // 使用转换器插件，从NV12 转换到 nvosd 所需要的RGBA
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  // 创建一个gstreamer element, 类型为nvdsosd，名称为nv-onscreendisplay。
  // 创建OSD在转换后的RGBA缓冲区上绘图
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  // 判断设备平台是否为TEGRA,是的话创建transform元件，实现渲染osd输出。
  // 这个属性是在makefile文件中设置的。
#ifdef PLATFORM_TEGRA
  transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
  // 接收器元件
  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

  // 确认各个元件均已创建
  if (!source || !h264parser || !decoder || !pgie
      || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

#ifdef PLATFORM_TEGRA
  if(!transform) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif

```

数据源元件负责从磁盘文件中读取视频数据，它具有名为 location 的属性，用来指明文件在磁盘上的位置。使用标准的**GObject**属性机制可以为元件设置相应的属性：

```
 // 使用命令行参数中的第二个参数（本地视频地址）设为source元件的location属性赋值
 g_object_set (G_OBJECT (source), "location", argv[1], NULL);
```

同理，为流多路复用器元件中的属性赋值:

```
  // 为streammux 元件中的batch-size属性赋值为1,表示只有一个数据源
  g_object_set (G_OBJECT (streammux), "batch-size", 1, NULL);

  // 为streammux 元件中的width、height、batched-push-timeout属性赋值
  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
  MUXER_OUTPUT_HEIGHT,"batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

```

同理，为推理元件的属性 config-file-path 赋值。推理中的属性可以在 dstest1_pgie_config.txt 配置文件中修改。

```
// 设置nvinfer元件中的属性config-file-path， 通过设置配置文件来设置nvinfer元件的所有必要属性
  g_object_set (G_OBJECT (pgie),
      "config-file-path", "dstest1_pgie_config.txt", NULL);

```

得到管道的消息总线：

```
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
```

添加消息监控器：

```
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);
```

此时。管道、元件都已经创建并赋值。现在需要把创建好的元件按照顺序，需要全部添加到管道中。

```
// 设置好管道，将所有元件添加到管道中。根据PLATFORM_TEGRA属性来决定是否将transform加入到管道中
#ifdef PLATFORM_TEGRA
  gst_bin_add_many(GST_BIN(pipeline),
  source, h264parser, decoder, streammux, pgie,
  nvvidconv, nvosdm, transform, sink, NULL);
#else
  gst_bin_add_many(GST_BIN(pipeline),
  source, h264parser, decoder, streammux, pgie,
  nvvidconv, nvosdm, sink, NULL);
#endif
```

现在，我们需要通过**pad**来将元件连接起来。**pad**是一个 element 的输入/输出**接口**，分为 src pad（生产）和 sink pad（消费）两种。在 element 通过 pad 连接成功后，数据会从上一个 element 的 src pad 传到下一个 element 的 sink pad，然后进行处理。

```
  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);  // 消费pad
  if (!sinkpad){
  	g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad (decoder, pad_name_src); // 生产pad
  if (!srcpad) {
    g_printerr ("Decoder request src pad failed. Exiting.\n");
    return -1;
  }

  // 将创建好的元件按照顺序连接起来，decoder -> streammux（生产 -> 消费）
  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
      return -1;
  }

  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  // 将创建好的元件按照顺序连接起来，source -> h264parser -> decoder
  if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
      g_printerr ("Elements could not be linked: 1. Exiting.\n");
      return -1;
  }

  // 将创建好的元件按照顺序连接起来，streammux -> pgie -> nvvidconv -> nvosd -> video -> sink
#ifdef PLATFORM_TEGRA
  if (!gst_element_link_many (streammux, pgie, nvvidconv, nvosd, transform, sink, NULL)) {
  	g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#else
  if (!gst_element_link_many (streammux, pgie,
      nvvidconv, nvosd, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#endif
```

添加探测来获得生成的元数据的信息：添加探测到 osd 元素的接收单元，因为到那时，缓冲区已经获得了所有的元数据。  
osd_sink_pad_buffer_probe 函数的作用就是获取到所有元数据信息，在此基础上画框和打印文字。

```
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink"); // 生产衬垫
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL); // 添加处理
  gst_object_unref (osd_sink_pad);
```

所有准备工作都做好之后，就可以通过将管道的状态切换到**PLAYING**状态，来启动整个管道的数据处理流程：

```
  g_print ("Now playing: %s\n", argv[1]);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);
```

进入主训练, 等待管道遇到错误或者 EOS 而终止

```
  g_print ("Running...\n");
  g_main_loop_run (loop);
```

跳出循环，终止管道，并释放资源

```
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
```

#### osd_sink_pad_buffer_probe 函数

这个函数的作用：提取从 osd 接收器接收到的元数据，并更新绘图举行，对象信息等的参数。  
DeepStream 中的数据结构，可以阅读官方文档：https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_metadata.html

```
static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL; // 目标检测元数据类型变量
    guint vehicle_count = 0; // 车辆数量
    guint person_count = 0; // 行人数量
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;  // 展示的元数据类型变量

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);  // 函数 gst_buffer_get_nvds_batch_meta()作用是 从 Gst Buffer 中提取 NvDsBatchMeta

    // 对batch_meta 中的frame_meta_list进行遍历; frame_meta_list列表的长度是num_frames_in_batch
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        // 对单帧的obj_meta_list进行遍历
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            // 判断目标检测框的类别如果是0（0是汽车的类别标签），那么车辆数+1，目标检测框数+1.
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            // 判断目标检测框的类别如果是2（0是行人的类别标签），那么行人数+1，目标检测框数+1.
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        // 设置display_meta的text_params属性
        NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        // 设置文本在画面中的x,y坐标（分别是相对于画面原点的偏移量）
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        // 设置文本的字体类型、字体颜色、字体大小
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        // 设置文本的背景颜色
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        // 将display_meta 信息添加到frame_meta信息中
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    g_print ("Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_number, num_rects, vehicle_count, person_count);
    frame_number++;
    return GST_PAD_PROBE_OK;
}
```

DeepStream 中的数据结构：  
![](https://pic1.zhimg.com/80/v2-e1107d5d3947dcbd9d38004ba54b16d8_720w.jpg)

#### bus_call 函数

消息处理函数，用于监视产生的消息。

```
static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}
```

### 运行效果

#### 命令

- 1、检查 deepstream 是否安装成功：

```
deepstream-app --version-all
```

- 2、工程目录下执行 make，编译，编译完后生成 deepstream-test1-app
- 3、运行：
  ![](https://pic1.zhimg.com/80/v2-8defb9802dd97d9db8fb032151c95390_720w.png)

#### 效果

![](https://pic1.zhimg.com/80/v2-9dc108ab72aa53f806a8bc737e3cc29c_720w.jpg)

### 参考

DeepStream SDK 开发指南：https://docs.nvidia.com/metropolis/deepstream/dev-guide/

DeepStream 概况： https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html

DeepStream 数据结构：https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_metadata.html

GStreamer 学习笔记： https://www.cnblogs.com/phinecos/archive/2009/06/07/1498166.html
