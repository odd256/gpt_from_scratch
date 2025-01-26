# 关键点

- 在gpt2中，attn block可以被看作一个reduce的过程，MLP可以看作Map的过程
- lm_head的bias=false，减少模型参数数量，隐藏状态经过层层计算，已经包含了丰富的信息
- 如何保证输入的信息是 masked
- 分词是如何实现的
- 输出时是如何变成句子的，beam search 有没有用上？
- 如何实现token缓存
- 什么是 test-time-inference
