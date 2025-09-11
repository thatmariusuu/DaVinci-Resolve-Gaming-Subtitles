# Welcome to Your DaVinci Resolve Automation Toolkit! ✨

## 📺 Video Tutorial

> **Disclaimer:** This is my first tutorial and it's a bit rushed because I was excited to share this milestone with you all!

**[🎬 Watch the Complete Setup & Usage Tutorial](https://youtu.be/lwDoL0rssA0)**  
*Learn how to install, configure, and use all the tools in this toolkit with step-by-step video guidance.*

> **Note:** Both this GitHub repository and the video tutorial represent the current state of the toolkit. While the repository will be maintained for major bug fixes, the UI and functionalities shown in the video are locked as they are at this moment.

---

Hello, fellow creator! Ready to make your editing process a whole lot smoother? This repository is packed with a suite of powerful Python scripts designed to bring advanced automation and AI capabilities right into your DaVinci Resolve workflow.

Whether you're crafting a documentary, producing content for YouTube, or editing a feature film, these tools are here to help you reclaim your time and focus on what you do best: telling amazing stories.

## What's Inside? 🚀

This toolkit is designed to feel like a natural extension of DaVinci Resolve, handling the tedious tasks so you can focus on your creative vision.

- **🤖 AI-Powered Transcription**: Get incredibly accurate, word-level timestamped transcripts thanks to a custom implementation of **WhisperX**, optimized for processing multiple smaller clips efficiently.
- **🎤 Crystal-Clear Vocals**: With **Demucs** integration, you can isolate dialogue from background noise, ensuring your transcripts are clean and precise, even with challenging audio.
- **🎬 Automated Interview Editor**: Magically create dynamic "interview-style" edits! The script intelligently cuts between speakers based on who is talking, saving you hours of manual cutting.
- **✨ Smart Gap-Filling**: Go beyond simple cuts! This feature fills silent moments in one speaker's track with relevant reactions or dialogue from another, creating a natural and engaging conversational flow.
- **✏️ Pro-Level Subtitles**: Generate subtitles that are not only accurate but also beautifully formatted. Using **spaCy** for linguistic analysis, subtitles are split at natural breaking points for perfect readability.
- **🎮 Dynamic Gaming-Style Subtitles**: Import your subtitles into Resolve with advanced logic that creates an entertaining, dynamic, and engaging visual style, perfect for YouTube and other online content.
- **🖥️ User-Friendly Interfaces**: All tools feature intuitive graphical interfaces (GUIs) built with PyQt5 and Tkinter, making them easy and enjoyable to use directly within DaVinci Resolve.
- **📁 Flexible Timeline Control**: By using the **OpenTimelineIO** format, the scripts allow for powerful, non-destructive timeline editing.
- **🎨 Ready-to-Use Templates**: Includes pre-configured DaVinci Resolve project files and XML presets for immediate use.

## Your Creative Toolkit 🧰

This repository includes three main scripts that work together in a seamless workflow, plus essential DaVinci Resolve files:

### 1. The Automated Interview Editor (`Interview Style Speech Timeline Generator.py`)

This is your starting point. This tool is designed for editors working with interview or dialogue-heavy content across multiple audio tracks.

- **What it does**:
  - Analyzes multiple audio tracks and automatically creates a multi-camera style edit.
  - Use **Enhanced Interview Mode** to intelligently fill silent gaps for a seamless conversation.
  - Exports a new timeline file (`.otio`) that you can use in the next step.
  - Utilizes the included XML preset files for audio rendering from DaVinci Resolve.

### 2. The Ultimate Transcription Suite (`whisper subtitles in turns...`)

Once you have your `.otio` files, this is your all-in-one command center for transcription.

- **What it does**:
  - A beautiful GUI to manage your transcription projects.
  - **Listens to your `.otio` files** and generates highly accurate SRT subtitle files from the speech.
  - Use the **Professional Transcription Editor** to compare and perfect your transcripts.

### 3. The Subtitle Importer & Styler (`ResolveSubtitlePro.py`)

This is the final step, where you bring your subtitles into Resolve and give them a dynamic, professional look.

- **What it does**:
  - Imports your SRT files into DaVinci Resolve.
  - Applies complex logic to create an **entertaining, gaming-video style** presentation for your subtitles.
  - Uses your favorite **Text+ templates** for maximum customization.
  - Works seamlessly with the included snap captions templates.

### 4. DaVinci Resolve Files Collection 📁

The `Davinci Resolve Files` folder contains everything you need to get started immediately:

- **`Davinci Resolve Project.drp`**: A complete DaVinci Resolve project file containing professionally designed snap captions templates for subtitles. Simply open this project file in DaVinci Resolve to access the pre-built subtitle templates. **This is used by the Subtitle Importer & Styler script** to provide precise positioning and styling for your imported subtitles.

- **XML Audio Presets** (`AudioOnly1.xml` through `AudioOnly4.xml`): These are essential render preset files that enable the Automated Interview Editor script to export specific audio tracks from DaVinci Resolve for processing. Each preset corresponds to a specific audio track:
  - `AudioOnly1.xml`: Exports only Audio Track 1 from your timeline
  - `AudioOnly2.xml`: Exports only Audio Track 2 from your timeline
  - `AudioOnly3.xml`: Exports only Audio Track 3 from your timeline
  - `AudioOnly4.xml`: Exports only Audio Track 4 from your timeline

## Getting Started 🛠️

Let's get you set up and ready to create!

### Prerequisites

- **DaVinci Resolve**: Download and install from the [Blackmagic Design website](https://www.blackmagicdesign.com/support/family/davinci-resolve-and-fusion). Look for the **"Latest Downloads"** section and download **DaVinci Resolve version 20.1.1 or any version 20.x**. This toolkit was developed and tested with DaVinci Resolve 20.1.1, but should work with any DaVinci Resolve 20.x version as demonstrated in the video tutorial.
- **Python 3.12**: This toolkit requires Python 3.12. Download and install from the [official Python website](https://www.python.org/downloads/) or use this direct link for Windows 64-bit: [Python 3.12.10](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe).
- **Visual Studio Code (Recommended)**: A powerful, free code editor that's perfect for editing Python scripts and managing your projects. Here's how to set it up:
  
  1. **Download and Install VS Code**: Go to the [official VS Code website](https://code.visualstudio.com/download) and download the installer for your operating system (Windows, macOS, or Linux). Run the installer and follow the setup wizard.
  
  2. **Install the Python Extension**: After launching VS Code for the first time:
     - Click on the Extensions icon in the left sidebar (or press `Ctrl+Shift+X` on Windows/Linux or `Cmd+Shift+X` on macOS)
     - Search for "Python" in the extensions marketplace
     - Install the official **Python extension by Microsoft** (it should be the first result with millions of downloads)
     - This extension provides syntax highlighting, IntelliSense code completion, debugging, code formatting, and integrated terminal support
  
  3. **Configure Python Interpreter**: Once the Python extension is installed:
     - Open any Python file (`.py`) or create a new one
     - VS Code will automatically detect your Python installation, or you can manually select it by pressing `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS) and typing "Python: Select Interpreter"
     - Choose the Python interpreter where you installed the required packages from the one-line installation command below
  
  With this setup, you'll have a professional development environment for editing and running the Python scripts in this toolkit!
- **FFmpeg**: This is a crucial open-source tool for handling video and audio files. The scripts rely on it to process media. It must be installed and accessible from your system's command line.

- **On Windows:**
  1. **Easy Installation (Recommended)**: Open Command Prompt or PowerShell as Administrator and run:
     ```cmd
     winget install FFmpeg
     ```
     This uses Windows' built-in package manager (available on Windows 10/11) and automatically handles PATH configuration.
  
  2. **Alternative Package Managers**: If you have [Chocolatey](https://chocolatey.org/) or [Scoop](https://scoop.sh/) installed, you can also use:
     ```cmd
     choco install ffmpeg
     ```
     or
     ```powershell
     scoop install ffmpeg
     ```
  
  3. **Manual Installation (If above methods don't work)**: Download a pre-built version of FFmpeg from a trusted source like [gyan.dev](https://www.gyan.dev/ffmpeg/builds/). The "essentials" build is usually sufficient.
  4. Extract the downloaded `.7z` file (you may need a tool like 7-Zip).
  5. Move the extracted folder to a permanent location (e.g., `C:\ffmpeg`).
  6. Add the `bin` directory inside that folder (e.g., `C:\ffmpeg\bin`) to your system's `PATH` environment variable. This allows you to run `ffmpeg` from any command prompt.
  7. To verify, open a new Command Prompt and type `ffmpeg -version`. You should see version information printed.

  - **On macOS:**
    The easiest way to install FFmpeg is with [Homebrew](https://brew.sh/). If you don't have Homebrew, install it first. Then, open your Terminal and run:
    ```bash
    brew install ffmpeg
    ```

  - **On Linux:**
    You can install FFmpeg using your distribution's package manager. Open your terminal and run the appropriate command:
    - **Debian/Ubuntu:**
      ```bash
      sudo apt update && sudo apt install ffmpeg
      ```
    - **Fedora/CentOS/RHEL:**
      ```bash
      sudo dnf install ffmpeg
      ```
    - **Arch Linux:**
      ```bash
      sudo pacman -S ffmpeg
      ```

### Installation Instructions

Choose the appropriate installation method based on your system:

---

## 🖥️ **FOR SYSTEMS WITHOUT NVIDIA GPU** 🖥️

### **⚡ QUICK INSTALL - NO GPU REQUIRED**

Open your terminal or command prompt and run this command:

```bash
set TEMP=C:\tmp && set TMP=C:\tmp && pip install ctranslate2 && pip install faster-whisper && pip install nltk && pip install onnxruntime && pip install pandas && pip install transformers && pip install numpy && pip install librosa && pip install torch && pip install torchvision && pip install torchaudio && pip install tqdm && pip install matplotlib && pip install demucs && pip install pipwin && pipwin install pyaudio && pip install pydub && pip install PyQt5 && pip install huggingface-hub && pip install opentimelineio && pip install pyannote-audio --no-deps
```

**✅ That's it! You're done with installation.**

---

## 🚀 **FOR SYSTEMS WITH NVIDIA GPU** 🚀

### **Step 1: Install Base Packages**

First, run this command:

```bash
set TEMP=C:\tmp && set TMP=C:\tmp && pip install ctranslate2 && pip install faster-whisper && pip install nltk && pip install onnxruntime && pip install pandas && pip install transformers && pip install numpy && pip install librosa && pip install tqdm && pip install matplotlib && pip install demucs && pip install pyaudio && pip install pydub && pip install PyQt5 && pip install huggingface-hub && pip install opentimelineio && pip install pyannote-audio --no-deps
```

### **Step 2: Continue with GPU Setup**

Then continue with the GPU acceleration setup below.

### GPU Acceleration with NVIDIA CUDA (Optional, but Recommended)

For a significant speed boost in transcription and audio processing, you can configure the scripts to use your NVIDIA GPU. This requires installing the NVIDIA CUDA Toolkit and the correct version of PyTorch.

**Step 1: Install NVIDIA Drivers**

Ensure you have the latest NVIDIA drivers installed for your GPU. You can download them from the [official NVIDIA website](https://www.nvidia.com/Download/index.aspx).

**Step 2: Install PyTorch with CUDA Support**

After installing the NVIDIA drivers, you need to install a version of PyTorch that is compiled for your specific CUDA version.

1.  Go to the [PyTorch installation page](https://pytorch.org/get-started/locally/).
2.  Use the interactive tool to generate the correct installation command for your system. Here is an example of the selections:
    *   **PyTorch Build:** Stable (e.g., 2.7.1)
    *   **Your OS:** Windows/Mac/Linux
    *   **Package:** Pip
    *   **Language:** Python
    *   **Compute Platform:** Select the CUDA version you want to use (e.g., CUDA 12.8).
3.  The website will generate a command for you to run. It will look similar to this:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Important:** Run the exact command generated by the PyTorch website. This will replace the version of PyTorch you installed earlier with one that can communicate with your GPU.

**NOTE:** This toolkit requires Python 3.12, which is fully compatible with the latest PyTorch versions.

### Setting Up DaVinci Resolve Files

**Step 1: Import the Project Template**
1. Open DaVinci Resolve
2. Go to `File -> Import Project`
3. Navigate to the `Davinci Resolve Files` folder and select `Davinci Resolve Project.drp`
4. This will load a complete project with pre-configured snap captions templates for subtitles

**Step 2: Install Audio Render Presets**
1. In DaVinci Resolve, go to `File -> Import -> Render Preset`
2. Navigate to the `Davinci Resolve Files` folder
3. Import the XML preset files (`AudioOnly1.xml` through `AudioOnly4.xml`) one by one
4. These presets will now be available in your render settings and are essential for the Interview Style Speech Timeline Generator script to properly render audio from DaVinci Resolve

### Script Setup

1.  **For the Transcription Suite**:
    - You can run the `whisper subtitles in turns...` script from anywhere on your computer. Just launch it, and it will connect with DaVinci Resolve.

2.  **For the Resolve-Integrated Tools**:
    - To make the `Interview Style Speech Timeline Generator.py` and `ResolveSubtitlePro.py` scripts appear inside DaVinci Resolve, copy them to the following directory:
      - **Windows**: `C:\ProgramData\Blackmagic Design\DaVinci Resolve\Fusion\Scripts\Utility\`
      - **macOS**: `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/`
      - **Linux**: `~/.local/share/BlackmagicDesign/DaVinci Resolve/Fusion/Scripts/Utility/`

    - After copying the files, restart DaVinci Resolve. You'll find the scripts under the `Workspace -> Scripts -> Utility` menu.

## Your Workflow: From Timeline to Final Subtitles

Here's how you'll use these tools together to create amazing content:

**Step 1: Set Up Your Templates**
- Import the `Davinci Resolve Project.drp` file to access pre-built snap captions templates
- Import the XML audio render presets (`AudioOnly1.xml` through `AudioOnly4.xml`) for optimal audio processing

**Step 2: Create Your Automated Edit**
- In DaVinci Resolve, run the **`Interview Style Speech Timeline Generator.py`** script from the `Workspace -> Scripts -> Utility` menu
- Select the audio tracks for your speakers and configure your settings
- The script will use the imported XML presets to render audio from DaVinci Resolve for processing
- The script will generate one or more `.otio` files, which represent your new, automatically created edit

**Step 3: Transcribe Your Edit**
- Run the **`whisper subtitles in turns...`** script
- Load the `.otio` files you created in the previous step
- The tool will "listen" to the edits in the `.otio` files and generate accurate SRT subtitle files
- Use the built-in editor to review and perfect the transcription

**Step 4: Add and Style Your Subtitles**
- Go back into DaVinci Resolve and run the **`ResolveSubtitlePro.py`** script
- Select the SRT files you just generated
- The script will import the subtitles and apply its dynamic, gaming-style formatting using the snap captions templates from the imported project file

## Let's Create! 🎬

You're all set! I hope these tools inspire you and help you bring your creative projects to life more efficiently than ever before. With the included DaVinci Resolve project file and XML presets, you can start creating professional-quality content right away.

If you have any questions or ideas, feel free to contribute!

Happy editing!