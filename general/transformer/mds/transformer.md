课程视频链接：  
[台大李宏毅自注意力机制和Transformer详解！](https://www.bilibili.com/video/BV1v3411r78R?p=4&spm_id_from=333.880.my_history.page.click)

课件链接：  
[seq2seq_v9.pdf](../imgs/seq2seq_v9.pdf)  
[self-attension_v7.pdf](../imgs/self-attension_v7.pdf)

# Sequence-to-sequence 场景

输入一个序列，输出一个序列，且输出序列的长度由模型决定。常见场景：语音识别、语音翻译、机器翻译、语音生成、聊天机器人、QA。 
![seq2seq.png](../imgs/seq2seq.png)   
![seq2seq_TTS.png](../imgs/seq2seq_TTS.png)   
![seq2seq_chatbot.png](../imgs/seq2seq_chatbot.png)   
![seq2seq_QA.png](../imgs/seq2seq_QA.png)   

## Seq2seq for Multi-label Classification  

一个目标可能有多个标签，但不知道有多少个，这种情况下就可以让Seq2seq模型决定标签的数量（序列）。
![seq2seq_multi-label.png](../imgs/seq2seq_multi-label.png)   

# Sequence-to-sequence 模型

模型结构：  
![seq2seq_model.png](../imgs/seq2seq_model.png)  

## Encoder
Encoder:  
![Encoder.png](../imgs/Encoder.png)  
Block:  
![Block.png](../imgs/Block.png)  
对输入的向量进行self-attention操作，每个向量都考虑与其它向量的关联性之后然后输出，然后接入到FC中输出结果。
Self-attension:  
![Self-attension.png](../imgs/Self-attension.png)  

### layer norm

![LN.png](../imgs/LN.png)  


![LN_BN.png](../imgs/LN_BN.png)  

BatchNorm是对一个batch-size样本内的每个特征做归一化，LayerNorm是对每个样本的所有特征做归一化。  
形象点来说，假设有一个二维矩阵。行为batch-size，列为样本特征。那么BN就是竖着归一化，LN就是横着归一化。  

它们的出发点都是让该层参数稳定下来，避免梯度消失或者梯度爆炸，方便后续的学习。但是也有侧重点。  
一般来说，如果你的特征依赖于不同样本间的统计参数，那BN更有效。因为它抹杀了不同特征之间的大小关系，但是保留了不同样本间的大小关系。（CV领域）  
而在NLP领域，LN就更加合适。因为它抹杀了不同样本间的大小关系，但是保留了一个样本内不同特征之间的大小关系。对于NLP或者序列任务来说，一条样本的不同特征，其实就是时序上字符取值的变化，样本内的特征关系是非常紧密的。


BERT的网络结构与transformer的Encoder相同：  
![BERT.png](../imgs/BERT.png)  

## Decoder
![Decoder.png](../imgs/Decoder.png)  

### Decoder-Autoregressive(AT)  
以语音识别为例：  
![Decoder_AT.png](../imgs/Decoder_AT.png)   

Decoder上一节点的output作为下一节点的输入（regressive），永不停止。因此要加入stop token来停止（token 标记）。  

![stop_token.png](../imgs/stop_token.png)   

Decoder的输入从一个start token开始，每一个输出位的输出可能是整个汉字集合，做一个softmax来决定到底输出哪个汉字。  

最终：  

![AT.png](../imgs/AT.png)   

#### masked self-attention

why masked？  
1. 加入masked attention后，transformer的decoder在功能上其实相当于rnn了，当前输出只与当前和过去输入有关，而与未来信息无关，二者区别在于，rnn的历史信息，只能一级一级的传递到当前时间步，而decoder直接使用attention，可以直接实现信息传递，比如，t-4时刻的信息，rnn只能传播4次才能到t时刻，attention只需要传播一次即可。
2. 训练阶段：训练时计算loss是用当前decoder输入所有单词对应位置的输出y1,y2,...yt与gt去算cross entropy loss，然后把t个loss加起来。如果用self-attention，则y1这个输出是用到x1右侧的单词信息的（用到未来的信息），而实际推理过程中显然是不可能提前知道未来信息的。
3. 预测阶段：预测结果是迭代生成的；预测阶段要保持重复的单词预测结果是一样的，这样不仅合理，而且可以增量更新（我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中）；与训练时的模型架构保持一致，前向传播的方式是一致的。

![重复单词预测结果一样.png](../imgs/重复单词预测结果一样.png)   



Encoder与Decoder中attention的不同：  

![masked_multi-head_attention.png](../imgs/masked_multi-head_attention.png)   

self-attention类似全连接层，输出向量的每一位与输入向量的每一位都有关，即b1与a1~a4都有关：  

![self-attention.png](../imgs/self-attention.png)   
![Self-attention2.png](../imgs/self-attention2.png)   
![self-attension3.png](../imgs/self-attension3.png)   


masked self-attention，类似LSTM，当前输出与之前的输入有关，与之后的输入无关，即b2与a1，a2有关，与a3、a4无关： 
![masked_self_attention.png](../imgs/masked_self_attention.png)   
![masked_self_attention2.png](../imgs/masked_self_attention2.png)   




### Decoder-Non Autoregressive(NAT)  


![AT_NAT.png](../imgs/AT_NAT.png)   


NAT每一个输入都是start，不考虑上一节点的输出。  
优点：并行，更稳定的生成（例如TTS，语音生成，只考虑当前词）  
缺点：性能一般比AT差，原因在于Multi-modality   

终止：   
1. 单独设置一个预测器进行终止预测；  
2. 输出一个非常长的序列，忽略End后面的tokens。  
   



## Encodet to Decoder

![cross_attention.png](../imgs/cross_attention.png)  

corss attention，将decoder的输入和encoder的输入进行交叉注意力计算。  
![cross_attention2.png](../imgs/cross_attention2.png)  

### training

![train_loss.png](../imgs/train_loss.png)  
![teacher_forcing.png](../imgs/teacher_forcing.png)  



