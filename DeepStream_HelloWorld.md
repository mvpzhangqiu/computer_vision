# DeepStream
deepstream-test DeepStream的hello world  

![](https://pic1.zhimg.com/80/v2-19572211c2bc6e891e55935eeaae6b84_720w.jpg) 

##  test1

### pipeline

首先数据源元件负责从磁盘上**读取**视频数据，解析器元件负责对数据进行**解析**，编码器元件负责对数据进行**解码**，流多路复用器元件负责**批处理**帧以实现最佳推理性能，推理元件负责实现**加速推理**，转换器元件负责将数据格式**转换**为输出显示支持的格式，可视化元件负责将边框与文本等信息**绘制**到图像中，渲染元件和接收器元件负责**输出**到屏幕上。  
![](https://pic1.zhimg.com/80/v2-f178b1cef853a6d19a2d2b079bb0f680_720w.jpg)  

### code

#### main函数

首先定义了需要用到的所有变量。因为所有GStreamer元件都具有相同的基类**GstElement**，因此能够采用GstElement类型对所有的元件进行定义。以及定义了负责数据消息的传输的GstBus类别变量。  

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
然后在主函数中调用gst_init()来完成相应的初始化工作，以便将用户从命令行输入的参数传递到GStreamer函数库。  
```
gst_init (&argc, &argv);
```
创建主循环，在执行g_main_loop_run后正式开始循环  
```
loop = g_main_loop_new (NULL, FALSE);
```
在GStreamer框架中，管道是用来容纳和管理元件的，下面创建一条名为pipeline的管道：
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
数据源元件负责从磁盘文件中读取视频数据，它具有名为location的属性，用来指明文件在磁盘上的位置。使用标准的**GObject**属性机制可以为元件设置相应的属性： 
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
同理，为推理元件的属性config-file-path赋值。推理中的属性可以在dstest1_pgie_config.txt配置文件中修改。 
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



















