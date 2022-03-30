# Deep-learning-homework-1

## 文件介绍
- Network.py 为网络模型
- Training.py 用于**训练**出12个不同**模型**
- load_data.py 用于读取数据
- load_netwotk.py 用于**参数查找**与**测试模型**并可视化参数
- main.py 用于训练某个特定模型并绘制loss与accuracy曲线

## 运行注意事项
- 训练与测试时，需要训练集与测试集数据放在data文件夹
- 测试时，需要模型参数文件放在model文件夹

## 训练步骤
- 若想训练出12个模型，请执行以下指令。
```python Training.py```
- 若想用于训练某个特定模型（若不更改文件，则为：alpha=0.001，hidden=100，Lambda=0.001模型）并绘制loss与accuracy曲线，请执行以下指令。
```python main.py```

## 测试步骤
- 若想测试训练出来的模型，请执行以下指令，并按照指示输入模型文件名称。
```python load_model.py```

## 模型文件
https://drive.google.com/drive/folders/1Yl6H1DfXmdn0E13HHiplJNTuIAGL1qja?usp=sharing
