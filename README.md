# Welcome to Your DaVinci Resolve Automation Toolkit! ✨

Hello, fellow creator! Ready to make your editing process a whole lot smoother? This repository is packed with a suite of powerful Python scripts designed to bring advanced automation and AI capabilities right into your DaVinci Resolve workflow.

Whether you're crafting a documentary, producing content for YouTube, or editing a feature film, these tools are here to help you reclaim your time and focus on what you do best: telling amazing stories.

## What's Inside? 🚀

This toolkit is designed to feel like a natural extension of DaVinci Resolve, handling the tedious tasks so you can focus on your creative vision.

- **🤖 AI-Powered Transcription**: Get incredibly accurate, word-level timestamped transcripts thanks to a custom implementation of **WhisperX**, optimized for processing multiple smaller clips efficiently.
- **🎤 Crystal-Clear Vocals**: With **Demucs** integration, you can isolate dialogue from background noise, ensuring your transcripts are clean and precise, even with challenging audio.
- **🎬 Automated Interview Editor**: Magically create dynamic "interview-style" edits! The script intelligently cuts between speakers based on who is talking, saving you hours of manual cutting.
- **✨ Smart Gap-Filling**: Go beyond simple cuts! This feature fills silent moments in one speaker's track with relevant reactions or dialogue from another, creating a natural and engaging conversational flow.
- **✍️ Pro-Level Subtitles**: Generate subtitles that are not only accurate but also beautifully formatted. Using **spaCy** for linguistic analysis, subtitles are split at natural breaking points for perfect readability.
- **🎮 Dynamic Gaming-Style Subtitles**: Import your subtitles into Resolve with advanced logic that creates an entertaining, dynamic, and engaging visual style, perfect for YouTube and other online content.
- **🖥️ User-Friendly Interfaces**: All tools feature intuitive graphical interfaces (GUIs) built with PyQt5 and Tkinter, making them easy and enjoyable to use directly within DaVinci Resolve.
- **📁 Flexible Timeline Control**: By using the **OpenTimelineIO** format, the scripts allow for powerful, non-destructive timeline editing.

## Your Creative Toolkit 🧰

This repository includes three main scripts that work together in a seamless workflow:

### 1. The Automated Interview Editor (`Interview Style Speech Timeline Generator.py`)

This is your starting point. This tool is designed for editors working with interview or dialogue-heavy content across multiple audio tracks.

- **What it does**:
  - Analyzes multiple audio tracks and automatically creates a multi-camera style edit.
  - Use **Enhanced Interview Mode** to intelligently fill silent gaps for a seamless conversation.
  - Exports a new timeline file (`.otio`) that you can use in the next step.

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

## Getting Started 🛠️

Let's get you set up and ready to create!

### Prerequisites

- **DaVinci Resolve**: Download and install from the [Blackmagic Design website](https://www.blackmagicdesign.com/products/davinciresolve/).
- **Python**: Download and install from the [official Python website](https://www.python.org/downloads/).
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

### One-Line Installation

Open your terminal or command prompt and run this single command to install all the necessary Python libraries:

```bash
pip install ctranslate2 faster-whisper nltk numpy onnxruntime pandas pyannote-audio torch torchaudio transformers
```

### GPU Acceleration with NVIDIA CUDA (Optional, but Recommended)

For a significant speed boost in transcription and audio processing, you can configure the scripts to use your NVIDIA GPU. This requires installing the NVIDIA CUDA Toolkit and the correct version of PyTorch.

**Step 1: Install NVIDIA Drivers**

Ensure you have the latest NVIDIA drivers installed for your GPU. You can download them from the [official NVIDIA website](https://www.nvidia.com/Download/index.aspx).

**Step 2: Install the CUDA Toolkit**

You will need to install the NVIDIA CUDA Toolkit. We recommend using **CUDA 12.1**. You can find the installer for your operating system in the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

**Step 3: Install PyTorch with CUDA Support**

After installing the CUDA Toolkit, you need to install a version of PyTorch that is compiled for your specific CUDA version.

1.  Go to the [PyTorch installation page](https://pytorch.org/get-started/locally/).
2.  Use the interactive tool to generate the correct installation command for your system. Here is an example of the selections:
    *   **PyTorch Build:** Stable (e.g., 2.7.1)
    *   **Your OS:** Windows/Mac/Linux
    *   **Package:** Pip
    *   **Language:** Python
    *   **Compute Platform:** Select the CUDA version you installed (e.g., CUDA 12.8).
3.  The website will generate a command for you to run. It will look similar to this:

    ```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```

    **Important:** Run the exact command generated by the PyTorch website. This will replace the version of PyTorch you installed earlier with one that can communicate with your GPU.

    **NOTE:** Latest PyTorch requires Python 3.9 or later.

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

Here’s how you’ll use these tools together to create amazing content:

**Step 1: Create Your Automated Edit**
- In DaVinci Resolve, run the **`Interview Style Speech Timeline Generator.py`** script from the `Workspace -> Scripts -> Utility` menu.
- Select the audio tracks for your speakers and configure your settings.
- The script will generate one or more `.otio` files, which represent your new, automatically created edit.

**Step 2: Transcribe Your Edit**
- Run the **`whisper subtitles in turns...`** script.
- Load the `.otio` files you created in the previous step.
- The tool will "listen" to the edits in the `.otio` files and generate accurate SRT subtitle files.
- Use the built-in editor to review and perfect the transcription.

**Step 3: Add and Style Your Subtitles**
- Go back into DaVinci Resolve and run the **`ResolveSubtitlePro.py`** script.
- Select the SRT files you just generated.
- The script will import the subtitles and apply its dynamic, gaming-style formatting to them on your timeline.

## Let's Create! 🎬

You're all set! I hope these tools inspire you and help you bring your creative projects to life more efficiently than ever before. If you have any questions or ideas, feel free to contribute!

Happy editing!