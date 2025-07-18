# Fogsight (雾象) [**English**](./readme_en.md) | [**中文**](./readme.md)


<p align="center">
  <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/logos/fogsight_logo_white_bg.png"
       alt="Fogsight Logo"
       width="300">
</p>

**雾象是一款由大型语言模型（LLM）驱动的动画引擎 agent 。用户输入抽象概念或词语，雾象会将其转化为高水平的生动动画。**

将雾象部署在本地后，您只需输入词语，点击生成，便可得到动画。


<p align="center">
  <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/1.png"
       alt="UI 截图"
       width="550">
</p>

我们设计了易用的语言用户界面（Language User Interface），用户也可以**进一步轻松编辑或改进生成动画，做到言出法随**。

雾象，意为 **“在模糊智能中的具象”**。*雾象是 WaytoAGI 开源计划项目成员。 WaytoAGI， 让更多人因 AI 而强大*


## 动画示例

以下为 Fogsight AI 生成的动画示例，点击以跳转并查看


<table>
  <tr>
    <td align="center">
      <a href="https://www.bilibili.com/video/BV1PXgKzBEyN">
        <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/thumbnails/entropy_increase_thumbnail.png" width="350"><br>
        <strong>The Law of Increasing Entropy (Physics)</strong><br>
        <strong>熵增定律 (物理学)</strong><br>
        <em>输入: 熵增定律</em>
      </a>
    </td>
    <td align="center">
      <a href="https://www.bilibili.com/video/BV1yXgKzqE42">
        <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/thumbnails/euler_formula_thumbnail.png" width="350"><br>
        <strong>Euler's Polyhedron Formula (Mathematics)</strong><br>
        <strong>欧拉多面体定理 (数学)</strong><br>
        <em>输入: 欧拉定理</em>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://www.bilibili.com/video/BV1sQgKzMEox">
        <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/thumbnails/bubble_sort_thumbnail.png" width="350"><br>
        <strong>Bubble Sort (Computer Science)</strong><br>
        <strong>冒泡排序 (计算机科学)</strong><br>
        <em>输入: 冒泡排序</em>
      </a>
    </td>
    <td align="center">
      <a href="https://www.bilibili.com/video/BV1yQgKzMEo6">
        <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/thumbnails/affordance_thumbnail.png" width="350"><br>
        <strong>Affordance (Design)</strong><br>
        <strong>可供性 (设计学)</strong><br>
        <em>输入: affordance in design</em>
      </a>
    </td>
  </tr>
</table>

## 核心功能

* **概念即影像**: 输入一个主题，Fogsight 将为您生成一部叙事完整的高水平动画，包含双语旁白与电影级的视觉质感。  
* **智能编排**: Fogsight 的核心是其强大的LLM驱动的编排能力。从旁白、视觉元素到动态效果，AI 将自动完成整个创作流程，一气呵成。  
* **语言用户界面 (LUI)**: 通过与 AI 的多轮对话，您可以对动画进行精准调优和迭代，直至达到您心中最理想的艺术效果。  

## 快速上手

### 环境要求

* Python 3.9+  
* 一个现代网络浏览器 (如 Chrome, Firefox, Edge)  
* 大语言模型的 API 密钥。我们推荐您使用 Google Gemini 2.5，但当前版本仅提供 OpenAI SDK 格式，您可以使用 OpenRouter、One API 或其他中转服务来调用它。  

### 安装与运行

1. **克隆代码仓库:**
   ```bash
   git clone https://github.com/macnversion/fogsight.git
   cd fogsight
   ```

2. **安装依赖:**

   ```bash
   pip install -r requirements.txt
   ```

3. **配置API密钥:**

   ```bash
   cp demo-credentials.json credentials.json
   # 复制 demo-credentials.json 文件并重命名为 credentials.json
   # 编辑 credentials.json 文件，设置 active_provider 并填入对应的 API_KEY 和 BASE_URL
   # 支持多个 LLM 提供商：Gemini、OpenAI、Claude、Kimi、Ark 等
   # 只需修改 active_provider 字段即可切换不同的 LLM 服务
   ```

4. **一键启动:**

   ```bash
   python start_fogsight.py
   # 运行 start_fogsight.py 脚本
   # 它将自动启动后端服务并在浏览器中自动打开 http://127.0.0.1:8000
   ```

5. **开始创作！**
   在页面中输入一个主题（例如“冒泡排序”），然后等待结果生成。

## 联系我们/加入群聊

请访问[此链接](https://fogsightai.feishu.cn/wiki/WvODwyUr1iSAe0kEyKfcpqvynGc?from=from_copylink)联系我们或加入交流群。

## Contributors

### 高校

* [@taited](https://taited.github.io/) - 香港中文大学（深圳） 博士生
* [@yjydya](https://github.com/ydyjya) - 南洋理工大学 博士生

### WaytoAGI 社区

* [@richkatchen 陈财猫](https://okjk.co/enodyA) - WaytoAGI 社区成员
* [@kk](https://okjk.co/zC8myE) - WaytoAGI 社区成员

### Index Future Lab

* [何淋 (@Lin he)](https://github.com/zerohe2001)

### AI 探索家

* [黄小刀 (@Xiaodao Huang)](https://okjk.co/CkFav6)

### 独立开发者与 AI 艺术家

* [@shuyan-5200](https://github.com/shuyan-5200)
* [王如玥 (@Ruyue Wang)](https://github.com/Moonywang)
* [@Jack-the-Builder](https://github.com/Jack-the-Builder)
* [@xiayurain95](https://github.com/xiayurain95)
* [@Lixin Cai 蔡李鑫](https://github.com/Lixin-Cai)

## 开源许可

本项目基于 MIT 许可证开源。
不过，如果您愿意在引用本项目时加上我们的署名与指向本项目的链接，我们会非常感谢 😊。
