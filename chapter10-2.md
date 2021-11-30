---
marp: true
---
<style>
h3 {
    color: black;
    text-align: center;
}
h4 {
    color: black;
    position: absolute;
    top: 30px;
    left: 80px;
    border-left: 10px solid #00A784;
    padding-left: 30px;
}
h5 {
    background-image: url("images/logo.png");
    background-size: 180px;
    background-repeat:no-repeat;
    color: black;
    position: absolute;
    top: 5px;
    right: 30px;
    height: 60px;
    padding-left: 180px;
    padding-top: 10px;
    font-size: 0px;
    font-weight: normal;
}
</style>
![bg left:50% height:500px](images/nangua.jpg)
<style scoped>
h1 {
  color: black;
  text-align: center;
}
h2 {
  color: black;
  text-align: center;
}
p {
    margin-top: 50px;
    color: black;
    text-align: center;
}
</style>
# 《机器学习公式详解》<br>（南瓜书）
## 第10章 降维和度量学习(下)
##### 异步社区
本节主讲：秦州

---
#### 本节大纲
##### 异步社区
南瓜书对应章节：10.5 10.6

1.流形学习
 - 等度量映射
 - 局部线性嵌入
 
2.度量学习

---
#### 流形学习(manifold learning)
##### 异步社区
“流形”：在局部具有欧式空间的性质。
虽然样本在高维空间中的分布看起来非常复杂，但是只要它们在局部上仍然具有欧式空间的性质，因此可以在局部建立降维映射关系，然后再设法将局部映射关系推广到全局。

---
#### 等度量映射(Isometric Mapping)
##### 异步社区
保持近邻样本之间的距离，因为测地线距离和高危空间的直线距离是不相等的
![w:1000](./images/ISO.png)




---
#### 等度量映射2
##### 异步社区
利用流形在局部上与欧式空间同胚的性质，找出每个点的欧式近邻，建立近邻的链接图，计算两个点之间的测地线距离转化为计算近邻链接图上两点之间的最短路径问题。
![w:950](./images/isomap_algo.png)

---
#### 等度量映射3
##### 异步社区
1.近邻图如何构建？
- $k$近邻图：指定$k$个点为近邻点  
- $\epsilon$近邻图：指定距离阈值$\epsilon$  
2. 对新样本如何变换
- 构建高维坐标和低维坐标的回归器


---
#### 局部线性嵌入(Locally Linear Embedding)
##### 异步社区
保持邻域内样本之间的线性关系。即假设样本点$\boldsymbol{x}_i$的坐标能通过它的邻域样本$\boldsymbol{x}_j,\boldsymbol{x}_k,\boldsymbol{x}_l$的坐标通过线性组合而重构得到：
$$
\boldsymbol{x}_{i}=w_{i j} \boldsymbol{x}_{j}+w_{i k} \boldsymbol{x}_{k}+w_{i l} \boldsymbol{x}_{l}
$$
![w:700](./images/LLE.png)




---
#### 局部线性嵌入
##### 异步社区

假设$\mathbf{Z}$的第$i$行记作$\mathbf{Z}_{i\cdot}$ 该行的均值记作$\bar{\boldsymbol{z}}$则
$$\sum^{d'}_{i=1}\frac{1}{m}\left(\mathbf{Z}_{i\cdot}-\bar{\boldsymbol{z}}\right)\left(\mathbf{Z}_{i\cdot}-\bar{\boldsymbol{z}}\right)^\mathrm{T}=\frac{1}{m}\sum^{d'}_{i=1}\mathbf{Z}_{i\cdot}\mathbf{Z}^\mathrm{T}_{i\cdot}=\frac{1}{m}\mathrm{tr}\left(\mathbf{Z}\mathbf{Z}^\mathrm{T}\right)
$$
根据之前定义的记号：$\mathbf{Z}=\mathbf{W}^{\mathrm{T}} \mathbf{X}$
忽略常数项，优化目标可写为
$$
\begin{array}{ll}
\max_{\mathbf{W}} & \operatorname{tr}\left(\mathbf{W}^{\mathrm{T}} \mathbf{X} \mathbf{X}^{\mathrm{T}} \mathbf{W}\right) \\\\
\text { s.t. } & \mathbf{W}^{\mathrm{T}} \mathbf{W}=\mathbf{I}
\end{array}
$$

---
#### 主成分分析-求解
##### 异步社区
使用拉格朗日乘子法，写出拉格朗日函数
$$
L(\mathbf{W}, \boldsymbol{\Lambda})=-\operatorname{tr}\left(\mathbf{W}^{\top} \mathbf{X X}^{\top} \mathbf{W}\right)+\left(\mathbf{W}^{\top} \mathbf{W}-\mathbf{I}\right) \boldsymbol{\Lambda}
$$
其中，
$$
\Lambda=\left[\begin{array}{cccc}
\lambda_{1} & & & \\
& \lambda_{2} & & \\
& & \ddots & \\
& & & \lambda_{d^{\prime}}
\end{array}\right] \in \mathbb{R}^{d^{\prime} \times d^{\prime}}, \mathbf{I}=\left[\begin{array}{cccc}
1 & & & \\
& 1 & & \\
& & \ddots & \\
& & & 1
\end{array}\right] \in \mathbb{R}^{d^{\prime} \times d^{\prime}}
$$




--- 
#### 主成分分析-求解2
##### 异步社区

对$\mathbf{W} \in \mathbb{R}^{d \times d^{\prime}}$求导：
$$
\begin{aligned}
\frac{\partial L(\mathbf{W}, \boldsymbol{\Lambda})}{\partial \mathbf{W}} &=-\frac{\partial \operatorname{tr}\left(\mathbf{W}^{\top} \mathbf{X} \mathbf{X}^{\top} \mathbf{W}\right)}{\partial \mathbf{W}}+\frac{\partial\left(\mathbf{W}^{\top} \mathbf{W}-\mathbf{I}\right)}{\partial \mathbf{W}} \boldsymbol{\Lambda} \\
&=-\mathbf{X X}^{\top} \mathbf{W}-\left(\mathbf{X X}^{\top}\right)^{\top} \mathbf{W}+2 \mathbf{W} \boldsymbol{\Lambda} \\
&=-2 \mathbf{X} \mathbf{X}^{\top} \mathbf{W}+2 \mathbf{W} \mathbf{\Lambda}
\end{aligned}
$$
另偏导$\frac{\partial L(\mathbf{W}, \boldsymbol{\Lambda})}{\partial \mathbf{W}}=0$，得：
$$
\mathbf{X X}^{\top} \mathbf{W}=\mathbf{W} \boldsymbol{\Lambda}
$$
或者将此式拆分成$d'$个式子：
$$
\mathbf{X X}^{\top} \boldsymbol{w}_{i}=\lambda_{i} \boldsymbol{w}_{i}, 1 \leqslant i \leqslant d
$$
即求矩阵$\mathbf{X X}^{\top} \in \mathbb{R}^{d \times d}$特征值和特征向量的形式。


---
#### 主成分分析-求解3
##### 异步社区
$\mathbf{X X}^{\top} \in \mathbb{R}^{d \times d}$有$d$个特征向量，但是我们只需要其中$d'$个，如何选择特征向量？
对 $\mathbf{X X}^{\top} \mathbf{W}=\mathbf{W} \boldsymbol{\Lambda}$ 两边同乘 $\mathbf{W}^{\top}$, 得
$$
\mathbf{W}^{\top} \mathbf{X X}^{\top} \mathbf{W}=\mathbf{W}^{\top} \mathbf{W} \mathbf{\Lambda}=\mathbf{\Lambda}
$$
我们的优化目标
$$
\operatorname{tr}\left(\mathbf{W}^{\top} \mathbf{X X}^{\top} \mathbf{W}\right)=\operatorname{tr}(\boldsymbol{\Lambda})=\sum_{i=1}^{d} \lambda_{i}
$$
因此最大化迹即选择最大的$d'$个$\lambda_i$，和它们所对应的$w_i$组成矩阵$\mathbf{W}$





---
#### 预告
##### 异步社区
下一节：降维和度量学习(下)
流形学习
度量学习
西瓜书对应章节：10.5 10.6

---
#### 结束语
##### 异步社区
<style scoped>
img {
  display: block;
  margin: 0 auto;
  width: 280px;
}
</style>
欢迎加入【南瓜书读者交流群】，我们将在群里进行答疑、勘误、本次直播回放、本次直播PPT发放、下次直播通知等最新资源发放和活动通知。
加入步骤：
1. 关注公众号【Datawhale】，发送【南瓜书】三个字获取机器人“小豚”的微信二维码
2. 添加“小豚”为微信好友，然后对“小豚”发送【南瓜书】三个字即可自动邀请进群

![qrcode](images/qrcode.jpeg)