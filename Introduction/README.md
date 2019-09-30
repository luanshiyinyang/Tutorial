# TensorFlow2简介
<img src='tf2.gif' alt='' />


## 简介
- TensorFlow是谷歌开源的一款深度学习框架，首次发布于2015年，TensorFlow2发布于2019年，而今，TensorFlow已被很多企业与创业公司广泛用于自动化工作任务和开发新系统，其在分布式训练支持、可扩展的生产和部署选项、多设备支持（如安卓）方面备受好评。
- TensorFlow使用数据流模型（即计算图）来描述计算过程，并将它们映射到了各种不同的硬件平台上，包括Linux、MacOS、Windows、Android和iOS等，从x86架构到ARM架构，从拥有单个或多个CPU的服务器到大规模GPU集群。凭借着统一的架构，TensorFlow可以跨越多种平台进行部署，显著地降低了机器学习系统的应用部署难度。
- 相比于对于新手不太友好的TensorFlow1.0，TensorFlow2.0采用了比较简易的新框架，并且将Keras收购为子模块，大大加强了集成度，减少了使用难度。


## 使用原因
- 现在，比较主流的深度学习框架有不少，但是最为广泛使用且有大型企业维护支持，衍生项目最多的还是TensorFlow和PyTorch，后者以简洁的API和编码机制短短2年用户激增。
- 如果说TensorFlow1.0让用户感觉在Python之上学习了一种新的语言（对于新手，这确实是一个“灾难”，基础的控制语句都不可以使用Python语法），那么TensorFlow2.0则像PyTorch那样，可以认为是一个Python化的框架。
- 之所以选择TensorFlow是基于多方面考虑的（主要对比TensorFlow1.0和PyTorch）。
  - 可视化训练过程方便
    - TensorFlow有一套Tensorboard配套库，能跟踪运行，可视化算图、模型等丰富功能。（PyTorch使用visdom，没有Tensorboard方便实用）
  - API简洁
    - 像写Python程序一样设计神经网络。（不需要TF1的复杂机制）
  - 生产部署方便
    - 可以直接使用TensorFlow serving 在TensorFlow中部署模型，这是一种使用了REST Client API的框架。（而PyTorch的部署需要借助Django或者Flask这样的PythonWeb框架）
- 最为关键的是，**相对于PyTorch这样还较为年轻的框架，TensorFlow的生态已经相当成熟和完整，如今又修改整合了API以及调整为动态图机制，不论是开发者还是科研人员，TensorFlow2.0都是不错的选择。**


## 新版本变化
- 说明
  - 相比于TF1，TF2的变化某种程度上是翻天覆地的。TensorFlow2.0在1.x的基础上进行了重新设计，针对提高使用者的开发效率，对API做了精简，删除了冗余的API并使之更加一致。同时由原来的静态计算图转为动态计算图优先，使用function而不是session执行计算图。
- API
  - 很多TensorFlow1.x的API在2.0中被删除或者改变位置，有不少原来的API被全新的API取代了。*官方提供了一个工具，可以将1.x的代码升级到2.0，不过不是很实用，很多时候需要人工修改。*
- 动态图机制
  - Eager execution（动态图机制）从TensorFlow1.8就加入了，但是作为可选操作，默认的一直是Graph execution（静态图机制）。TensorFlow2.0中Eager execution被设置为默认模式。该模式最大的好处在于**用户能够更轻松的编写和调试代码，可以使用原生的Python控制语句（原来的TF1只能使用TensorFlow封装的控制语句），大大降低了学习和使用TensorFlow的门槛**。在TensorFlow2.0中，图（Graph）和会话（Session）都是底层的实现，用户不需要过多关心。
- 全局变量
  - TensorFlow1.x依赖隐式全局命名空间。当我们调用`tf.Variable`创建变量时，该变量就会被放进默认的图中，即使我们忘记了指向它的Python变量，它也会留在那里。当我们想恢复这些变量时，我们必须知道该变量的名称，如果我们没法控制这些变量的创建，也就无法做到这点。TensorFlow 1.x中有各种机制旨在帮助用户再次找到他们所创建的变量，而在2.0中则取消了所有这些机制，支持默认的机制：跟踪变量。当我们不再用到创建的某个变量时，该变量就会被自动回收（这和Python的回收机制类似）。


## 安装
- 说明
  - 指定虚拟环境下使用pip安装。（建议venv或者conda环境）
- CPU版本
  - `pip install tensorflow==2.0.0rc1`
- GPU版本
  - `pip install tensorflow-gpu==2.0.0rc1`


## 补充说明
- 本文并没有提及TensorFlow2的更多编码使用上的细节，因为在使用之前至少应该先了解使用的框架有什么优势和劣势。
- 博客已经同步至我的[个人博客网站](https://luanshiyinyang.github.io/tensorflow2/2019/09/21/Introduction/)，欢迎访问查看最新文章。
- 如有错误或者疏漏之处，欢迎指正。