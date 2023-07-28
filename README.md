# PDF智能优化工具

## 简介

这是一个使用Python和PySide6构建的图形用户界面(GUI)程序，它的主要功能是优化PDF文件。

## 功能

- 选择一个或多个PDF文件进行处理。
- 提供多种优化选项，包括：
  - 漂白效果：使PDF页面的颜色更加均匀。
  - 去槽点：通过图像处理技术去除PDF中的噪声点。
  - 文字锐化：增强PDF中文字的清晰度。
  - AI优化：使用Real-ESRGAN模型进行图像超分辨率处理，提升图像质量。
  - 压缩文件：在保存图像时进行压缩，以减小文件大小。
- 可以选择页面大小（A4、A3或自动识别）。
- 可以选择页面方向（横向、竖向或自动识别）。
- 提供一个进度条显示处理进度。
- 在处理完成后显示一个消息提示框。
- 包含一个菜单项，可以查看关于PySide6和Real-ESRGAN的信息。

## 安装

这个程序需要安装以下Python库：

pip install PySide6 Pillow PyMuPDF opencv-python numpy basicsr realesrgan torchvision
并且需要有一个Real-ESRGAN的预训练模型文件`RealESRGAN-x4plus.pth`。

## 使用

直接运行脚本即可启动程序：
python zspdfai.py
