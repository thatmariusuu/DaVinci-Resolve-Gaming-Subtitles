import os
import warnings
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set environment variable to disable Intel SVML before importing anything else
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Fix encoding issues
os.environ["PYTHONIOENCODING"] = "utf-8"

# Suppress SpeechBrain warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import tempfile
import subprocess
import time
import shutil
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import re
import signal
import platform
import traceback
import gc
import json
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
import difflib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Windows-compatible timeout with threading
import concurrent.futures

try:
    import opentimelineio as otio
except ImportError:
    print("OpenTimelineIO not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opentimelineio"])
    import opentimelineio as otio

sys.path.insert(0, r'.\whisperX')
import whisperx

try:
    import demucs
except ImportError:
    print("Demucs not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "demucs"])
    import demucs

try:
    import vlc
    VLC_AVAILABLE = True
except ImportError:
    VLC_AVAILABLE = False
    print("python-vlc not found. Installing...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-vlc"])
        import vlc
        VLC_AVAILABLE = True
    except Exception as e:
        print(f"Failed to install python-vlc: {e}")
        VLC_AVAILABLE = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox, QShortcut,
    QProgressBar, QSpinBox, QGroupBox, QListWidget, QListWidgetItem,
    QTreeWidget, QTreeWidgetItem, QSplitter, QAbstractItemView, QCheckBox,
    QTextEdit, QSizePolicy, QDialog, QFrame, QRadioButton, QMenu, QDoubleSpinBox,
    QButtonGroup, QLineEdit, QSpacerItem, QStyle, QSlider, QTabWidget, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QIcon, QPalette, QKeySequence

def run_ffmpeg_util_with_debug(cmd, operation_name, file_path=None):
    # Format the command for logging
    cmd_str = " ".join(cmd)
    
    # Create a detailed log entry
    log_parts = [f"=== FFmpeg Util {operation_name} ==="]
    
    if file_path:
        # Add file information to the log if available
        log_parts.append(f"File: {file_path}")
        
        # Add file size if available and file exists
        if os.path.exists(file_path):
            try:
                file_size = os.path.getsize(file_path)
                log_parts.append(f"File size: {file_size/1024:.2f} KB")
            except:
                pass
    
    # Add the full command
    log_parts.append(f"Command: {cmd_str}")
    
    # Join all parts with newlines and print
    log_message = "\n".join(log_parts)
    print(log_message)
    
    # Start time for performance tracking
    start_time = time.time()
    
    # Execute the command
    try:
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        success = True
        elapsed = time.time() - start_time
        print(f"✓ FFmpeg Util {operation_name} completed successfully in {elapsed:.2f}s")
        return success, process.stdout, process.stderr
    except subprocess.CalledProcessError as e:
        success = False
        elapsed = time.time() - start_time
        print(f"✗ FFmpeg Util {operation_name} failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        print(f"Error output: {e.stderr.decode() if hasattr(e, 'stderr') else 'No stderr'}")
        return success, e.stdout if hasattr(e, 'stdout') else None, e.stderr if hasattr(e, 'stderr') else None

def get_audio_stream_count(file_path):
    """Get the number of audio streams in the media file with detailed debugging."""
    cmd = [
        "ffprobe", 
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        file_path
    ]
    
    try:
        print(f"Getting audio stream count for: {file_path}")
        success, stdout, stderr = run_ffmpeg_util_with_debug(cmd, "audio stream count", file_path)
        
        if not success or not stdout:
            print(f"Failed to get audio stream count for: {file_path}")
            return 0
            
        data = json.loads(stdout)
        
        # Count audio streams
        audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
        
        print(f"Found {len(audio_streams)} audio streams in {file_path}")
        
        # Print details about each audio stream
        for i, stream in enumerate(audio_streams):
            codec = stream.get("codec_name", "unknown")
            channels = stream.get("channels", "?")
            sample_rate = stream.get("sample_rate", "?")
            bit_rate = stream.get("bit_rate", "?")
            actual_index = stream.get("index", i)
            print(f"  Stream {i} (Index: {actual_index}): {codec}, {channels} channels, {sample_rate} Hz, {bit_rate} bps")
            
        return len(audio_streams)
    except Exception as e:
        print(f"Error getting audio streams: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 0

def format_timecode(seconds):
    """Format seconds into a timecode string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    frames = int((seconds % 1) * 24)  # Assuming 24 fps
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

class TextComparisonUtils:
    """Enhanced text comparison utilities with smart punctuation handling"""
    
    # Ending punctuation priority (higher index = higher priority)
    ENDING_PUNCTUATION_PRIORITY = {'.': 1, ';': 2, '!': 3, '?': 4}
    
    @staticmethod
    def are_texts_essentially_same(text1, text2):
        """Check if two texts are essentially the same - ENHANCED with repeated character handling"""
        if not text1 or not text2:
            return False
        
        if text1.strip() == text2.strip():
            return True
        
        # Enhanced normalization that handles repeated characters
        norm1 = TextComparisonUtils._normalize_for_comparison(text1)
        norm2 = TextComparisonUtils._normalize_for_comparison(text2)
        
        if norm1 == norm2:
            return True
        
        # NEW: Special handling for very short texts with repeated characters
        if len(norm1) <= 10 and len(norm2) <= 10:
            # For short texts, check if they're the same after removing repeated chars
            pattern1 = re.sub(r'(.)\1+', r'\1', norm1)
            pattern2 = re.sub(r'(.)\1+', r'\1', norm2)
            if pattern1 == pattern2:
                return True
        
        # Use difflib for similarity check with lower threshold for short similar texts
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Lower threshold for short texts that might be repeated characters
        threshold = 0.85 if max(len(norm1), len(norm2)) <= 10 else 0.95
        return similarity >= threshold

    @staticmethod
    def _normalize_for_comparison(text):
        """Normalize text for comparison purposes - removes punctuation differences and handles repeated characters"""
        import re
        
        normalized = text.lower().strip()
        
        # Handle common word variations
        normalized = re.sub(r'\bOK\b', 'okay', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bok\b', 'okay', normalized, flags=re.IGNORECASE)
        
        # NEW: Handle repeated characters (hmmmm -> hmm, nooooo -> noo, etc.)
        # Replace 3+ consecutive identical characters with just 2
        normalized = re.sub(r'(.)\1{2,}', r'\1\1', normalized)
        
        # Remove all punctuation except for word boundaries
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()
        
        return normalized

    @staticmethod
    def merge_similar_texts(text1, text2):
        """
        Merge two similar texts, preferring longer text when they're essentially the same
        
        Args:
            text1, text2: The texts to merge
            
        Returns:
            str: Merged text with preference for longer similar content
        """
        if not text1 or not text2:
            return text1 or text2
        
        # If texts are identical, return as-is
        if text1.strip() == text2.strip():
            return text1.strip()
        
        # NEW: If texts are essentially the same, prefer the longer one
        if TextComparisonUtils.are_texts_essentially_same(text1, text2):
            # When texts are essentially the same, prefer the longer one
            longer_text = text1 if len(text1.strip()) >= len(text2.strip()) else text2
            shorter_text = text2 if longer_text == text1 else text1
            
            print(f"🔄 Merging similar texts: '{shorter_text.strip()}' → '{longer_text.strip()}' (chose longer)")
            
            # Get the best ending punctuation from both texts
            best_ending = TextComparisonUtils._get_best_ending_punctuation(text1, text2)
            
            # Remove any ending punctuation from longer text and add the best ending
            import re
            base_without_ending = re.sub(r'[.;!?]+$', '', longer_text.strip()).strip()
            
            if best_ending:
                merged_text = base_without_ending + best_ending
            else:
                merged_text = base_without_ending
            
            return merged_text
        
        # If not essentially the same, use original logic
        # Choose the text with more internal punctuation as the base
        internal_punct1 = TextComparisonUtils._count_internal_punctuation(text1)
        internal_punct2 = TextComparisonUtils._count_internal_punctuation(text2)
        
        if internal_punct1 >= internal_punct2:
            base_text = text1.strip()
            alt_text = text2.strip()
        else:
            base_text = text2.strip()
            alt_text = text1.strip()
        
        # Get the best ending punctuation from both texts
        best_ending = TextComparisonUtils._get_best_ending_punctuation(text1, text2)
        
        # Remove any ending punctuation from base text and add the best ending
        import re
        base_without_ending = re.sub(r'[.;!?]+$', '', base_text).strip()
        
        if best_ending:
            merged_text = base_without_ending + best_ending
        else:
            merged_text = base_without_ending
        
        return merged_text

    @staticmethod
    def _count_internal_punctuation(text):
        """Count punctuation marks that are not at the end of the text"""
        import re
        
        # Remove ending punctuation and count remaining punctuation
        text_without_ending = re.sub(r'[.;!?]+$', '', text.strip())
        internal_punct = re.findall(r'[^\w\s]', text_without_ending)
        return len(internal_punct)
    
    @staticmethod
    def _get_best_ending_punctuation(text1, text2):
        """Get the highest priority ending punctuation from two texts"""
        import re
        
        # Extract ending punctuation from both texts
        ending1 = TextComparisonUtils._extract_ending_punctuation(text1)
        ending2 = TextComparisonUtils._extract_ending_punctuation(text2)
        
        # Get priority scores
        priority1 = TextComparisonUtils.ENDING_PUNCTUATION_PRIORITY.get(ending1, 0)
        priority2 = TextComparisonUtils.ENDING_PUNCTUATION_PRIORITY.get(ending2, 0)
        
        # Return the one with higher priority
        if priority1 >= priority2:
            return ending1
        else:
            return ending2
    
    @staticmethod
    def _extract_ending_punctuation(text):
        """Extract the last punctuation mark from text"""
        import re
        
        if not text:
            return ""
        
        # Find the last character that's punctuation
        text = text.strip()
        if text and text[-1] in '.;!?':
            return text[-1]
        return ""
    
    @staticmethod
    def group_segments_by_time(all_segments):
        """Group segments that overlap in time - SINGLE IMPLEMENTATION"""
        if not all_segments:
            return []
        
        sorted_segments = sorted(all_segments, key=lambda x: x['start'])
        groups = []
        current_group = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            overlaps = False
            for group_seg in current_group:
                if (segment['start'] < group_seg['end'] and 
                    segment['end'] > group_seg['start']):
                    overlaps = True
                    break
            
            if overlaps:
                current_group.append(segment)
            else:
                groups.append(current_group)
                current_group = [segment]
        
        if current_group:
            groups.append(current_group)
        
        return groups

    @staticmethod
    def segment_group_has_differences(segment_group):
        """Check if a segment group has meaningful differences - ENHANCED with merging"""
        if len(segment_group) <= 1:
            return False
        
        texts = [seg['text'].strip() for seg in segment_group if seg['text'].strip()]
        
        if len(texts) <= 1:
            return False
        
        # Check if any texts are NOT essentially the same
        first_text = texts[0]
        for text in texts[1:]:
            if not TextComparisonUtils.are_texts_essentially_same(first_text, text):
                return True
        
        return False
    
    @staticmethod
    def get_best_merged_text_from_group(segment_group):
        """
        Get the best merged text from a group of segments that are essentially the same
        
        Args:
            segment_group: List of segments with similar text
            
        Returns:
            str: The best merged text with optimal punctuation
        """
        if not segment_group:
            return ""
        
        if len(segment_group) == 1:
            return segment_group[0]['text'].strip()
        
        # Get all texts
        texts = [seg['text'].strip() for seg in segment_group if seg['text'].strip()]
        
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # Start with the first text and merge with others
        merged_text = texts[0]
        for text in texts[1:]:
            if TextComparisonUtils.are_texts_essentially_same(merged_text, text):
                merged_text = TextComparisonUtils.merge_similar_texts(merged_text, text)
        
        return merged_text

class ModelComparisonResult:
    """Simplified - just stores data, minimal processing"""
    
    def __init__(self, clip_info, base_model_result, comparison_results):
        self.clip_info = clip_info
        self.base_model = base_model_result["model"]
        self.base_segments = base_model_result["segments"]
        self.comparison_results = comparison_results
        
        # Just store the data - let other classes handle complex analysis
        self.all_segments = self._combine_all_segments()
        self.segment_groups = TextComparisonUtils.group_segments_by_time(self.all_segments)
        
    def _combine_all_segments(self):
        """Simple segment combination"""
        all_segments = []
        
        # Add base model segments
        for seg in self.base_segments:
            all_segments.append({
                'model': self.base_model,
                'start': seg.start,
                'end': seg.end,
                'text': seg.text.strip(),
                'is_base': True
            })
        
        # Add comparison model segments
        for comp_result in self.comparison_results:
            for seg in comp_result["segments"]:
                all_segments.append({
                    'model': comp_result["model"],
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text.strip(),
                    'is_base': False
                })
        
        return all_segments
    
    def has_meaningful_differences(self):
        """Quick check if there are any meaningful differences"""
        for group in self.segment_groups:
            if TextComparisonUtils.segment_group_has_differences(group):
                return True
        return False

class ModelComparisonDialog(QDialog):
    """Enhanced Model Comparison Dialog with audio playback, inline editing, and professional editing features"""
    
    def __init__(self, comparison_results, parent=None):
        """Enhanced Model Comparison Dialog with audio playback, inline editing, and professional editing features"""
        super().__init__(parent)
        self.comparison_results = comparison_results
        self.selected_transcriptions = {}  # Store selected transcriptions per clip
        self.undo_stack = []  # Undo/redo stack
        self.redo_stack = []
        self.max_undo_operations = 50
        self.current_audio_file = None
        self.inline_edit_item = None
        
        # Initialize audio player as None first
        self.audio_player = None
        self.setup_segment_timer()

        # Collect all unique models from all results for consistent column ordering
        all_models = set()
        for result in self.comparison_results:
            all_models.add(result.base_model)
            for comp_result in result.comparison_results:
                all_models.add(comp_result["model"])
        
        # Sort models for consistent ordering and store for later use
        self.model_columns = sorted(list(all_models))
        
        self.setup_ui()
        self.setup_keyboard_shortcuts()
        self.populate_results()

    def setup_ui(self):
        """Set up the enhanced UI with audio player and professional controls"""
        self.setWindowTitle("Enhanced Model Comparison Results - Professional Transcription Editor")
        self.setMinimumSize(1600, 900)
        self.resize(1800, 1000)
        self.setModal(True)
        
        # Make window maximizable and resizable
        self.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        layout = QVBoxLayout(self)
        
        # Header with clip count
        header_label = QLabel(f"Enhanced Transcription Editor ({len(self.comparison_results)} clips)")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(header_label)
        
        # Toolbar
        self.setup_toolbar()
        layout.addWidget(self.toolbar)
        
        # Audio player section
        self.setup_audio_player()
        layout.addWidget(self.audio_group)
        
        # Search and filter section
        self.setup_search_filter()
        layout.addLayout(self.search_layout)
        
        # Main content area
        self.setup_main_content()
        layout.addWidget(self.main_content)
        
        # Status bar
        self.setup_status_bar()
        layout.addWidget(self.status_bar)
        
        # Control buttons
        self.setup_control_buttons()
        layout.addLayout(self.button_layout)

        # Update button states after all UI components are created
        self.update_button_states()

    def setup_segment_timer(self):
        """Set up segment timer for precise playback control - call this in __init__"""
        self.segment_timer = QTimer()
        self.segment_timer.timeout.connect(self.check_segment_end)
        self.segment_timer.setInterval(100)  # Check every 100ms
        
        # Initialize segment tracking variables
        self.current_segment_start = 0
        self.current_segment_end = 0
        self.playing_segment = None

    def check_segment_end(self):
        """Check if we've reached the end of the current segment"""
        if not self.audio_player or not hasattr(self, 'current_segment_end'):
            return
        
        try:
            # Get current playback position in seconds
            current_time_ms = self.audio_player.media_player.get_time()
            if current_time_ms < 0:  # VLC returns -1 if no media
                return
                
            current_time = current_time_ms / 1000.0
            
            # Check if we've passed the segment end
            if current_time >= self.current_segment_end:
                self.segment_timer.stop()
                
                # Check if looping is enabled
                if hasattr(self, 'loop_enabled') and self.loop_enabled and hasattr(self, 'playing_segment'):
                    # Restart the segment
                    self.play_specific_segment(self.playing_segment)
                else:
                    # Pause at segment end
                    self.audio_player.pause_playback()
                    if hasattr(self.audio_player, 'status_label'):
                        self.audio_player.status_label.setText("Segment finished")
                    
        except Exception as e:
            # Silently handle errors to avoid log spam
            pass

    def setup_toolbar(self):
        """Set up the main toolbar with common actions"""
        self.toolbar = QFrame()
        self.toolbar.setFixedHeight(50)
        self.toolbar.setStyleSheet("""
            QFrame {
                background-color: #ecf0f1;
                border-bottom: 1px solid #bdc3c7;
                border-radius: 5px;
            }
        """)
        
        toolbar_layout = QHBoxLayout(self.toolbar)
        
        # Playback controls
        self.play_button = QPushButton("▶")
        self.play_button.setFixedSize(40, 30)
        self.play_button.setToolTip("Play/Pause current segment (Space)")
        self.play_button.clicked.connect(self.spacebar_play_handler)

        self.stop_button = QPushButton("⏹")
        self.stop_button.setFixedSize(30, 30)
        self.stop_button.setToolTip("Stop playback")
        self.stop_button.clicked.connect(self.stop_playback)
        
        # Navigation controls
        self.prev_button = QPushButton("⏮")
        self.prev_button.setFixedSize(30, 30)
        self.prev_button.setToolTip("Previous segment (←)")
        self.prev_button.clicked.connect(self.go_to_previous_segment)

        self.next_button = QPushButton("⏭")
        self.next_button.setFixedSize(30, 30)
        self.next_button.setToolTip("Next segment (→)")
        self.next_button.clicked.connect(self.go_to_next_segment)
        
        # Separator
        separator1 = QFrame()
        separator1.setFrameStyle(QFrame.VLine | QFrame.Sunken)
        
        # Undo/Redo controls
        self.undo_button = QPushButton("↶")
        self.undo_button.setFixedSize(30, 30)
        self.undo_button.setToolTip("Undo (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo_last_action)
        
        self.redo_button = QPushButton("↷")
        self.redo_button.setFixedSize(30, 30)
        self.redo_button.setToolTip("Redo (Ctrl+Y)")
        self.redo_button.clicked.connect(self.redo_last_action)
        
        # Separator
        separator2 = QFrame()
        separator2.setFrameStyle(QFrame.VLine | QFrame.Sunken)
        
        # Delete control
        self.delete_button = QPushButton("🗑")
        self.delete_button.setFixedSize(30, 30)
        self.delete_button.setToolTip("Delete selected segments (Delete)")
        self.delete_button.clicked.connect(self.delete_selected_segments)
        
        # Separator
        separator3 = QFrame()
        separator3.setFrameStyle(QFrame.VLine | QFrame.Sunken)
        
        # Save/Export controls
        self.save_button = QPushButton("💾")
        self.save_button.setFixedSize(30, 30)
        self.save_button.setToolTip("Save progress (Ctrl+S)")
        self.save_button.clicked.connect(self.save_progress)
        
        self.export_button = QPushButton("📤")
        self.export_button.setFixedSize(30, 30)
        self.export_button.setToolTip("Export results")
        self.export_button.clicked.connect(self.export_results)
        
        # Add all controls to toolbar
        toolbar_layout.addWidget(self.play_button)
        toolbar_layout.addWidget(self.stop_button)
        toolbar_layout.addWidget(separator1)
        toolbar_layout.addWidget(self.prev_button)
        toolbar_layout.addWidget(self.next_button)
        toolbar_layout.addWidget(separator2)
        toolbar_layout.addWidget(self.undo_button)
        toolbar_layout.addWidget(self.redo_button)
        toolbar_layout.addWidget(separator3)
        toolbar_layout.addWidget(self.delete_button)
        toolbar_layout.addWidget(separator3)
        toolbar_layout.addWidget(self.save_button)
        toolbar_layout.addWidget(self.export_button)
        toolbar_layout.addStretch()

    def setup_keyboard_shortcuts(self):
        """Set up comprehensive keyboard shortcuts"""
        # Playback shortcuts
        QShortcut(Qt.Key_Space, self, self.spacebar_play_handler)  # CHANGED: Use new handler
        QShortcut(Qt.Key_Left, self, self.go_to_previous_segment)
        QShortcut(Qt.Key_Right, self, self.go_to_next_segment)

    def spacebar_play_handler(self):
        """Handle spacebar press - seek to content start and toggle playback (FIXED for 1s padding)"""
        if not self.audio_player:
            return
        
        # If we're not currently playing, seek to the content start first
        if not self.audio_player.is_playback_active():
            self.seek_to_current_segment_start()
            # Small delay to ensure seeking is complete before starting playback
            QTimer.singleShot(150, self.audio_player.start_playback)  # Increased delay slightly
        else:
            # If already playing, just pause
            self.audio_player.pause_playback()

    def save_progress(self):
        """Save current progress of the model comparison session"""
        from PyQt5.QtWidgets import QFileDialog
        import json
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Model Comparison Progress", 
            "model_comparison_progress.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Create a comprehensive save data structure
                save_data = {
                    "session_info": {
                        "timestamp": time.time(),
                        "total_clips": len(self.comparison_results),
                        "total_segments": self.main_content.topLevelItemCount(),
                        "undo_operations": len(self.undo_stack),
                        "manual_edits": sum(1 for clip_selections in self.selected_transcriptions.values()
                                          for selection in clip_selections.values() 
                                          if selection.get('is_manual', False))
                    },
                    "selected_transcriptions": {},
                    "clip_info": []
                }
                
                # Save selected transcriptions (convert objects to serializable format)
                for clip_idx, clip_selections in self.selected_transcriptions.items():
                    save_data["selected_transcriptions"][str(clip_idx)] = {}
                    for group_idx, selection in clip_selections.items():
                        save_data["selected_transcriptions"][str(clip_idx)][str(group_idx)] = {
                            "model": selection.get("model", ""),
                            "start": selection.get("start", 0),
                            "end": selection.get("end", 0),
                            "text": selection.get("text", ""),
                            "is_manual": selection.get("is_manual", False),
                            "is_edited": selection.get("is_edited", False)
                        }
                
                # Save basic clip information
                for i, result in enumerate(self.comparison_results):
                    clip_info = {
                        "index": i,
                        "name": result.clip_info.get('name', 'Unknown'),
                        "base_model": result.base_model,
                        "comparison_models": [comp["model"] for comp in result.comparison_results],
                        "total_segments": len(result.base_segments)
                    }
                    save_data["clip_info"].append(clip_info)
                
                # Write the save file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(
                    self, 
                    "Progress Saved", 
                    f"Model comparison progress saved to:\n{file_path}\n\n"
                    f"Saved {len(self.selected_transcriptions)} clip selections and {save_data['session_info']['manual_edits']} manual edits."
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save progress:\n{str(e)}")

    def setup_audio_player(self):
        """Set up the integrated audio player"""
        self.audio_group = QGroupBox("🎵 Audio Verification")
        self.audio_group.setFixedHeight(200)
        audio_layout = QVBoxLayout(self.audio_group)

        self.audio_player = AudioPlayerWidget()
        self.audio_player.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        audio_layout.addWidget(self.audio_player)
        
        # Connect audio player events
        self.audio_player.playback_stopped.connect(self.on_audio_stopped)

        # Audio position display
        self.audio_position_label = QLabel("00:00 / 00:00")
        self.audio_position_label.setAlignment(Qt.AlignCenter)
        self.audio_position_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        audio_layout.addWidget(self.audio_position_label)
        
    def setup_search_filter(self):
        """Set up search and filter controls"""
        self.search_layout = QHBoxLayout()
        
        # Search box
        search_label = QLabel("🔍 Find:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search transcriptions... (Ctrl+F)")
        self.search_box.textChanged.connect(self.apply_search_filter)
        self.search_box.setMaximumWidth(300)
        
        # Filter dropdown
        filter_label = QLabel("Filter:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "Show All Segments",
            "Show Only Differences", 
            "Show Only Significant Differences",
            "Show Only Manual Selections Needed"
        ])
        self.filter_combo.currentTextChanged.connect(self.apply_content_filter)
        self.filter_combo.setMaximumWidth(200)
        
        self.search_layout.addWidget(search_label)
        self.search_layout.addWidget(self.search_box)
        self.search_layout.addSpacing(20)
        self.search_layout.addWidget(filter_label)
        self.search_layout.addWidget(self.filter_combo)
        self.search_layout.addStretch()
        
    def setup_main_content(self):
        """Set up the main table with model columns"""
        self.main_content = QTreeWidget()
        
        self.main_content.setHeaderLabels([
            "Index", "Clip", "Time", "Actions"
        ])
        
        # Enhanced table features
        self.main_content.setAlternatingRowColors(True)
        self.main_content.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.main_content.setSortingEnabled(True)
        self.main_content.setContextMenuPolicy(Qt.CustomContextMenu)
        self.main_content.setEditTriggers(QAbstractItemView.DoubleClicked)
        
        # Connect signals
        self.main_content.itemClicked.connect(self.on_item_clicked)
        self.main_content.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.main_content.itemChanged.connect(self.on_item_changed)  # Track text changes
        self.main_content.itemSelectionChanged.connect(self.on_selection_changed)
        self.main_content.customContextMenuRequested.connect(self.show_context_menu)
        self.main_content.currentItemChanged.connect(self.on_current_item_changed)

    def on_item_changed(self, item, column):
        """Handle when item text is changed through editing - FIXED VERSION"""
        print(f"🐛 on_item_changed called: column={column}")
        
        if not hasattr(self, 'model_columns'):
            print("🐛 No model_columns, skipping")
            return
        
        # Check if this is a model column
        if column >= 3 and column < 3 + len(self.model_columns):
            print(f"🐛 This is a model column: {column}")
            
            # Check if this item/column is currently being edited
            edit_key = f"{id(item)}_{column}"
            if not hasattr(self, '_currently_editing'):
                self._currently_editing = set()
            
            was_being_edited = edit_key in self._currently_editing
            print(f"🐛 Was being edited: {was_being_edited}")
            
            if was_being_edited:
                # Remove from editing set
                self._currently_editing.discard(edit_key)
                
                # Get the new text and original text
                new_text = item.text(column).strip()
                original_text = item.data(column, Qt.UserRole + 10) or ""
                
                print(f"🐛 New text: '{new_text}', Original text: '{original_text}'")
                
                # Only process if text actually changed
                if new_text != original_text:
                    print("🐛 Text changed, calling handle_text_edit")
                    self.handle_text_edit(item, column, new_text, original_text)
                else:
                    print("🐛 Text unchanged, restoring original display")
                    # Restore the original display
                    self.restore_original_display(item, column, original_text)
            else:
                print("🐛 Item was not being edited, skipping")
        else:
            print(f"🐛 Not a model column (column {column}, model columns start at 3)")

    def restore_original_display(self, item, column, original_text):
        """Restore the original display text with proper formatting"""
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        
        # Check if this text was previously edited
        edited_texts = data.get("edited_texts", {})
        model_index = column - 3
        if model_index < len(self.model_columns):
            model_name = self.model_columns[model_index]
            
            if model_name in edited_texts:
                # This was previously edited, show with marker
                display_text = edited_texts[model_name]
                item.setText(column, display_text)
                # Keep the edited styling
                item.setBackground(column, QColor(255, 255, 200))
                font = item.font(column)
                font.setItalic(True)
                item.setFont(column, font)
            else:
                # Not edited, just show original text
                item.setText(column, original_text)

    def handle_text_edit(self, item, column, new_text, original_text):
        """Handle text edit with clean visual indicators only"""
        print(f"🐛 handle_text_edit called: column={column}, new_text='{new_text}'")
        
        # Save state for undo
        self.save_state_for_undo(f"Edit text in column {column}")
        
        # Get item data
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        
        # Get the model for this column
        model_index = column - 3
        if model_index >= len(self.model_columns) or model_index < 0:
            return
        
        edited_model = self.model_columns[model_index]
        
        # Clean text (no markers needed)
        clean_text = new_text.strip()
        
        # Store the edited text
        if "edited_texts" not in data:
            data["edited_texts"] = {}
        
        data["edited_texts"][edited_model] = clean_text
        item.setData(0, Qt.UserRole, data)
        
        # Update selected transcriptions if this model is currently selected
        clip_idx = data["clip_idx"]
        group_idx = data["group_idx"]
        selected_segment = data.get("selected")
        
        if selected_segment and selected_segment.get("model") == edited_model:
            updated_segment = selected_segment.copy()
            updated_segment["text"] = clean_text
            updated_segment["is_edited"] = True
            updated_segment["original_text"] = original_text
            
            data["selected"] = updated_segment
            item.setData(0, Qt.UserRole, data)
            self.selected_transcriptions[clip_idx][group_idx] = updated_segment
        
        # CLEAN VISUAL FEEDBACK - No text markers!
        # Just show the clean edited text with visual styling
        item.setText(column, clean_text)
        
        # Visual indicators for edited text
        item.setBackground(column, QColor(230, 255, 230))  # Light green background
        font = item.font(column)
        font.setItalic(True)  # Italic to show it's edited
        font.setBold(True)    # Bold to make it stand out
        item.setFont(column, font)
        
        # Optional: Add a subtle border or different text color
        item.setForeground(column, QColor(0, 120, 0))  # Dark green text
        
        self.update_status_display()

    def on_item_clicked(self, item, column):
        """Handle item clicks for model selection"""
        if not hasattr(self, 'model_columns'):
            return
        
        # Check if clicked column is a model column
        if column >= 3 and column < 3 + len(self.model_columns):
            model_index = column - 3
            selected_model = self.model_columns[model_index]
            
            # Get item data
            data = item.data(0, Qt.UserRole)
            if not data:
                return
            
            model_segments = data.get("model_segments", {})
            if selected_model in model_segments:
                # Select this model's transcription
                self.select_model_for_item(item, model_segments[selected_model], column)

    def on_item_double_clicked(self, item, column):
        """Handle double-clicks for text editing - IMPROVED VERSION"""
        if not hasattr(self, 'model_columns'):
            return
        
        # Check if double-clicked column is a model column
        if column >= 3 and column < 3 + len(self.model_columns):
            # Store the original text before editing
            original_text = item.text(column).strip()
            item.setData(column, Qt.UserRole + 10, original_text)  # Store original text
            
            # Mark this item/column combination as being edited
            edit_key = f"{id(item)}_{column}"
            if not hasattr(self, '_currently_editing'):
                self._currently_editing = set()
            self._currently_editing.add(edit_key)
            
            print(f"🐛 Starting edit for column {column}, original text: '{original_text}'")
            
            # Start editing the text
            item.setText(column, original_text)
            self.main_content.editItem(item, column)

    def setup_status_bar(self):
        """Set up the status bar with detailed information"""
        self.status_bar = QFrame()
        self.status_bar.setFixedHeight(60)
        self.status_bar.setStyleSheet("""
            QFrame {
                background-color: #34495e;    # This creates the blue-gray bar
                color: white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        status_layout = QHBoxLayout(self.status_bar)
        
        self.selected_count_label = QLabel("0 selected")
        self.selected_count_label.setStyleSheet("color: #3498db; font-weight: bold;")
        
        self.total_segments_label = QLabel("0 segments total")
        self.total_segments_label.setStyleSheet("color: #95a5a6;")
        
        self.current_audio_label = QLabel("No audio loaded")
        self.current_audio_label.setStyleSheet("color: #e67e22;")
        
        self.manual_edits_label = QLabel("0 manual edits")
        self.manual_edits_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        
        status_layout.addWidget(self.selected_count_label)
        status_layout.addWidget(QLabel("|"))
        status_layout.addWidget(self.total_segments_label)
        status_layout.addWidget(QLabel("|"))
        status_layout.addWidget(self.current_audio_label)
        status_layout.addWidget(QLabel("|"))
        status_layout.addWidget(self.manual_edits_label)
        status_layout.addStretch()

    def setup_control_buttons(self):
        """Set up the main control buttons"""
        self.button_layout = QHBoxLayout()
        
        # Selection helpers
        self.select_best_button = QPushButton("🎯 Auto-Select Best")
        self.select_best_button.setToolTip("Automatically select the best transcription for each segment")
        self.select_best_button.clicked.connect(self.auto_select_best)
        self.select_best_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        self.select_base_button = QPushButton("Select All Base Model")
        self.select_base_button.clicked.connect(self.select_all_base_model)
        
        # Main action button
        self.create_srt_button = QPushButton("✅ Create Final SRT Files")
        self.create_srt_button.clicked.connect(self.create_final_srt_files)
        self.create_srt_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        
        self.button_layout.addWidget(self.select_best_button)
        self.button_layout.addWidget(self.select_base_button)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.create_srt_button)
        self.button_layout.addWidget(self.close_button)

    def populate_results(self):
        """Populate the enhanced results table with model columns, filtering identical segments using centralized utilities"""
        self.main_content.clear()
        
        if not self.comparison_results:
            return
        
        # Set up headers with model columns
        headers = ["Index", "Clip", "Time"] + self.model_columns + ["Actions"]
        self.main_content.setHeaderLabels(headers)
        
        # Set column widths
        self.main_content.setColumnWidth(0, 60)   # Index
        self.main_content.setColumnWidth(1, 200)  # Clip name
        self.main_content.setColumnWidth(2, 100)  # Time
        
        # Set model column widths
        for i, model in enumerate(self.model_columns):
            col_index = 3 + i
            self.main_content.setColumnWidth(col_index, 200)
        
        # Actions column
        actions_col = 3 + len(self.model_columns)
        self.main_content.setColumnWidth(actions_col, 80)
        
        segment_index = 1
        total_segments = 0
        shown_segments = 0
        
        for clip_idx, result in enumerate(self.comparison_results):
            clip_name = result.clip_info.get("name", f"Clip {clip_idx + 1}")
            
            # Use the result's pre-computed segment groups (from refactored ModelComparisonResult)
            segment_groups = result.segment_groups
            
            # Initialize selections for this clip
            if clip_idx not in self.selected_transcriptions:
                self.selected_transcriptions[clip_idx] = {}
            
            for group_idx, segment_group in enumerate(segment_groups):
                total_segments += 1
                
                # Use centralized utility to check for differences
                if not TextComparisonUtils.segment_group_has_differences(segment_group):
                    # Auto-select the base model for hidden identical segments
                    base_segment = next((seg for seg in segment_group if seg.get('is_base', False)), segment_group[0])
                    self.selected_transcriptions[clip_idx][group_idx] = base_segment
                    continue  # Skip showing this segment
                
                shown_segments += 1
                
                # Create main segment item (only for segments with differences)
                segment_item = QTreeWidgetItem(self.main_content)
                
                # Set basic information
                segment_item.setText(0, f"{segment_index:03d}")
                segment_item.setText(1, clip_name)
                
                # Time information
                start_time = min(seg['start'] for seg in segment_group)
                end_time = max(seg['end'] for seg in segment_group)
                segment_item.setText(2, f"{start_time:.1f}-{end_time:.1f}s")
                
                # Create a mapping of model to segment text
                model_segments = {}
                for seg in segment_group:
                    model_segments[seg['model']] = seg
                
                # Default to base model selection
                base_segment = next((seg for seg in segment_group if seg.get('is_base', False)), segment_group[0])
                self.selected_transcriptions[clip_idx][group_idx] = base_segment
                
                # Fill in model columns
                for i, model in enumerate(self.model_columns):
                    col_index = 3 + i
                    if model in model_segments:
                        text = model_segments[model]['text']
                        segment_item.setText(col_index, text)
                        
                        # Make the cell editable
                        segment_item.setFlags(segment_item.flags() | Qt.ItemIsEditable)
                        
                        # Highlight selected model
                        if model_segments[model] == base_segment:
                            segment_item.setBackground(col_index, QColor(200, 255, 200))  # Light green
                            font = segment_item.font(col_index)
                            font.setBold(True)
                            segment_item.setFont(col_index, font)
                    else:
                        segment_item.setText(col_index, "")
                
                # Store data
                segment_item.setData(0, Qt.UserRole, {
                    "clip_idx": clip_idx,
                    "group_idx": group_idx,
                    "segments": segment_group,
                    "model_segments": model_segments,
                    "selected": base_segment,
                    "audio_file": result.clip_info.get("output_path")
                })
                
                # Add action button (delete)
                actions_col = 3 + len(self.model_columns)
                delete_btn = QPushButton("✕")
                delete_btn.setFixedSize(25, 25)
                delete_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #e74c3c;
                        color: white;
                        border: none;
                        border-radius: 12px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #c0392b;
                    }
                """)
                delete_btn.clicked.connect(lambda checked, item=segment_item: self.delete_segment(item))
                self.main_content.setItemWidget(segment_item, actions_col, delete_btn)
                
                segment_index += 1
        
        # Update status with filtering info
        hidden_count = total_segments - shown_segments
        if hidden_count > 0:
            self.log(f"Showing {shown_segments}/{total_segments} segments ({hidden_count} identical segments auto-selected)")
        
        self.update_status_display()

    def should_show_segment_group(self, segment_group):
        """Determine if a segment group should be shown (has meaningful differences)"""
        if len(segment_group) <= 1:
            return True  # Always show if only one model produced this segment
        
        # Get all unique texts from this group
        texts = []
        for seg in segment_group:
            text = seg['text'].strip()
            if text:  # Only consider non-empty texts
                texts.append(text)
        
        if len(texts) <= 1:
            return False  # All empty or only one non-empty text
        
        # Check if all texts are essentially the same
        first_text = texts[0]
        for text in texts[1:]:
            if not TextComparisonUtils.are_texts_essentially_same(first_text, text):
                return True  # Found a meaningful difference
        
        return False  # All texts are essentially the same

    def start_inline_editing(self, item, column):
        """Start inline editing for text columns"""
        if column == 4:  # Base text column
            self.inline_edit_item = item
            # The item is already editable, so Qt will handle the editing
            
    def start_inline_editing_current(self):
        """Start inline editing for the currently selected item"""
        current_item = self.main_content.currentItem()
        if current_item:
            self.main_content.editItem(current_item, 4)  # Edit base text column
            
    def cancel_inline_editing(self):
        """Cancel any ongoing inline editing"""
        if self.inline_edit_item:
            self.main_content.closePersistentEditor(self.inline_edit_item, 4)
            self.inline_edit_item = None
            
    def on_selection_changed(self):
        """Handle selection changes"""
        selected_items = self.main_content.selectedItems()
        
        # Auto-load audio for single selection
        if len(selected_items) == 1:
            self.load_audio_for_item(selected_items[0])
        
        self.update_status_display()
        self.update_button_states()
        
    def seek_to_current_segment_start(self):
        """Seek to the start of the current segment, accounting for 1-second padding"""
        if not self.audio_player:
            return
        
        current_item = self.main_content.currentItem()
        if current_item:
            data = current_item.data(0, Qt.UserRole)
            if data:
                segments = data.get("segments", [])
                if segments:
                    # Calculate the correct start position (accounting for 1-second padding)
                    start_time = min(seg['start'] for seg in segments)
                    seek_time = start_time
                    
                    # Seek to the position
                    self.audio_player.media_player.set_time(int(seek_time * 1000))
                    
                    # Update status
                    if hasattr(self.audio_player, 'status_label'):
                        self.audio_player.status_label.setText(f"Positioned at {seek_time:.1f}s (content start)")

    def on_current_item_changed(self, current, previous):
        """Handle current item changes for audio loading with padding consideration"""
        if current:
            self.load_audio_for_item(current)
            # Ensure we're positioned at the start of the new segment (after padding)
            QTimer.singleShot(300, self.seek_to_current_segment_start)  # Increased delay for reliability

    def load_audio_for_item(self, item):
        """Load audio file for the given item and auto-seek past padding"""
        if not self.audio_player:
            return
            
        data = item.data(0, Qt.UserRole)
        if not data:
            return
            
        audio_file = data.get("audio_file")
        if audio_file and os.path.exists(audio_file):
            if self.current_audio_file != audio_file:
                success = self.audio_player.load_audio_file(audio_file)
                if success:
                    self.current_audio_file = audio_file
                    filename = os.path.basename(audio_file)
                    self.current_audio_label.setText(f"♪ {filename}")
                    
                    # FIXED: Auto-seek to segment start + 1 second padding offset
                    segments = data.get("segments", [])
                    if segments:
                        start_time = min(seg['start'] for seg in segments)
                        seek_time = start_time
                        QTimer.singleShot(200, lambda: self.audio_player.media_player.set_time(int(seek_time * 1000)))

    def seek_to_time_with_padding(self, seek_time):
        """Helper method to seek to a specific time (already includes padding offset)"""
        if self.audio_player and self.audio_player.media_player:
            try:
                self.audio_player.media_player.set_time(int(seek_time * 1000))
                if hasattr(self.audio_player, 'status_label'):
                    self.audio_player.status_label.setText(f"Positioned at {seek_time:.1f}s")
            except Exception as e:
                print(f"Error seeking to time {seek_time}: {e}")

    def toggle_playback(self):
        """Toggle audio playback with automatic padding offset"""
        if not self.audio_player:
            return
        
        # If we're starting playback, ensure we're at the right position (after padding)
        if not self.audio_player.is_playback_active():
            current_item = self.main_content.currentItem()
            if current_item:
                data = current_item.data(0, Qt.UserRole)
                if data:
                    segments = data.get("segments", [])
                    if segments:
                        start_time = min(seg['start'] for seg in segments)
                        
                        # Set position before starting playback
                        self.audio_player.media_player.set_time(int(start_time * 1000))
        
        self.audio_player.toggle_playback()

    def stop_playback(self):
        """Stop audio playback"""
        if self.audio_player:
            self.audio_player.stop_playback()
            
    def go_to_previous_segment(self):
        """Navigate to previous segment"""
        current_item = self.main_content.currentItem()
        if current_item:
            index = self.main_content.indexOfTopLevelItem(current_item)
            if index > 0:
                prev_item = self.main_content.topLevelItem(index - 1)
                self.main_content.setCurrentItem(prev_item)
                self.main_content.scrollToItem(prev_item)
                
    def go_to_next_segment(self):
        """Navigate to next segment"""
        current_item = self.main_content.currentItem()
        if current_item:
            index = self.main_content.indexOfTopLevelItem(current_item)
            if index < self.main_content.topLevelItemCount() - 1:
                next_item = self.main_content.topLevelItem(index + 1)
                self.main_content.setCurrentItem(next_item)
                self.main_content.scrollToItem(next_item)
                
    def navigate_up(self):
        """Navigate up in the list"""
        self.go_to_previous_segment()
        
    def navigate_down(self):
        """Navigate down in the list"""
        self.go_to_next_segment()
        
    def go_to_first_segment(self):
        """Go to the first segment"""
        if self.main_content.topLevelItemCount() > 0:
            first_item = self.main_content.topLevelItem(0)
            self.main_content.setCurrentItem(first_item)
            self.main_content.scrollToItem(first_item)
            
    def go_to_last_segment(self):
        """Go to the last segment"""
        count = self.main_content.topLevelItemCount()
        if count > 0:
            last_item = self.main_content.topLevelItem(count - 1)
            self.main_content.setCurrentItem(last_item)
            self.main_content.scrollToItem(last_item)
            
    def page_up(self):
        """Page up in the list"""
        current_item = self.main_content.currentItem()
        if current_item:
            index = self.main_content.indexOfTopLevelItem(current_item)
            new_index = max(0, index - 10)  # Move up 10 items
            new_item = self.main_content.topLevelItem(new_index)
            self.main_content.setCurrentItem(new_item)
            self.main_content.scrollToItem(new_item)
            
    def page_down(self):
        """Page down in the list"""
        current_item = self.main_content.currentItem()
        if current_item:
            index = self.main_content.indexOfTopLevelItem(current_item)
            max_index = self.main_content.topLevelItemCount() - 1
            new_index = min(max_index, index + 10)  # Move down 10 items
            new_item = self.main_content.topLevelItem(new_index)
            self.main_content.setCurrentItem(new_item)
            self.main_content.scrollToItem(new_item)
            
    def focus_search(self):
        """Focus the search box"""
        self.search_box.setFocus()
        self.search_box.selectAll()
        
    def select_all_segments(self):
        """Select all segments"""
        self.main_content.selectAll()
        
    def delete_segment(self, item):
        """Delete a specific segment with undo support"""
        self.save_state_for_undo("Delete segment")
        
        # Mark as deleted (don't actually remove from tree for undo support)
        item.setHidden(True)
        data = item.data(0, Qt.UserRole)
        if data:
            data["deleted"] = True
            item.setData(0, Qt.UserRole, data)
            
        self.update_status_display()
        
    def delete_selected_segments(self):
        """Delete all selected segments"""
        selected_items = self.main_content.selectedItems()
        if not selected_items:
            return

        self.save_state_for_undo(f"Delete {len(selected_items)} segments")
        
        for item in selected_items:
            self.delete_segment(item)
                
    def save_state_for_undo(self, operation_name):
        """Save current state for undo functionality"""
        # Save current state to undo stack
        state = {
            "operation": operation_name,
            "timestamp": time.time(),
            "selections": self.selected_transcriptions.copy(),
            "hidden_items": []
        }
        
        # Save which items are hidden
        for i in range(self.main_content.topLevelItemCount()):
            item = self.main_content.topLevelItem(i)
            if item.isHidden():
                state["hidden_items"].append(i)
                
        self.undo_stack.append(state)
        
        # Limit undo stack size
        if len(self.undo_stack) > self.max_undo_operations:
            self.undo_stack.pop(0)
            
        # Clear redo stack when new operation is performed
        self.redo_stack.clear()
        
        self.update_button_states()
        
    def undo_last_action(self):
        """Undo the last action"""
        if not self.undo_stack:
            return
            
        # Save current state to redo stack
        current_state = {
            "operation": "Current state",
            "timestamp": time.time(),
            "selections": self.selected_transcriptions.copy(),
            "hidden_items": []
        }
        
        for i in range(self.main_content.topLevelItemCount()):
            item = self.main_content.topLevelItem(i)
            if item.isHidden():
                current_state["hidden_items"].append(i)
                
        self.redo_stack.append(current_state)
        
        # Restore previous state
        previous_state = self.undo_stack.pop()
        self.restore_state(previous_state)
        
        self.update_button_states()
        
    def redo_last_action(self):
        """Redo the last undone action"""
        if not self.redo_stack:
            return
            
        # Save current state to undo stack
        current_state = {
            "operation": "Redo checkpoint",
            "timestamp": time.time(),
            "selections": self.selected_transcriptions.copy(),
            "hidden_items": []
        }
        
        for i in range(self.main_content.topLevelItemCount()):
            item = self.main_content.topLevelItem(i)
            if item.isHidden():
                current_state["hidden_items"].append(i)
                
        self.undo_stack.append(current_state)
        
        # Restore redo state
        redo_state = self.redo_stack.pop()
        self.restore_state(redo_state)
        
        self.update_button_states()
        
    def restore_state(self, state):
        """Restore a saved state"""
        self.selected_transcriptions = state["selections"]
        
        # Restore hidden items
        for i in range(self.main_content.topLevelItemCount()):
            item = self.main_content.topLevelItem(i)
            should_be_hidden = i in state["hidden_items"]
            item.setHidden(should_be_hidden)
            
            # Update item data
            data = item.data(0, Qt.UserRole)
            if data:
                data["deleted"] = should_be_hidden
                item.setData(0, Qt.UserRole, data)
                
        self.update_status_display()
        
    def update_button_states(self):
        """Update toolbar button states"""
        has_undo = len(self.undo_stack) > 0
        has_redo = len(self.redo_stack) > 0
        has_selection = len(self.main_content.selectedItems()) > 0
        
        self.undo_button.setEnabled(has_undo)
        self.redo_button.setEnabled(has_redo)
        self.delete_button.setEnabled(has_selection)
        
        # Update tooltips with operation names
        if has_undo:
            last_op = self.undo_stack[-1]["operation"]
            self.undo_button.setToolTip(f"Undo: {last_op} (Ctrl+Z)")
        else:
            self.undo_button.setToolTip("Undo (Ctrl+Z)")
            
        if has_redo:
            last_redo = self.redo_stack[-1]["operation"]
            self.redo_button.setToolTip(f"Redo: {last_redo} (Ctrl+Y)")
        else:
            self.redo_button.setToolTip("Redo (Ctrl+Y)")
            
    def apply_search_filter(self, search_text):
        """Apply search filter to segments"""
        search_text = search_text.lower()
        
        for i in range(self.main_content.topLevelItemCount()):
            item = self.main_content.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            
            # Skip deleted items
            if data and data.get("deleted", False):
                continue
                
            # Search in all text columns
            visible = not search_text or any(
                search_text in item.text(col).lower() 
                for col in range(self.main_content.columnCount())
            )
            
            item.setHidden(not visible)
            
    def apply_content_filter(self, filter_type):
        """Apply content-based filter"""
        for i in range(self.main_content.topLevelItemCount()):
            item = self.main_content.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            
            if not data:
                continue
                
            # Skip deleted items unless showing all
            if data.get("deleted", False) and filter_type != "Show All":
                item.setHidden(True)
                continue
                
            visible = True
            
            if filter_type == "Show Differences Only":
                # Only show segments where models disagree
                segments = data["segments"]
                texts = set(seg['text'] for seg in segments)
                visible = len(texts) > 1
                
            elif filter_type == "Show Selected Only":
                # Only show segments with non-base selections
                selected = data["selected"]
                visible = not selected.get('is_base', False)
                
            elif filter_type == "Show Manual Edits":
                # Only show manually edited segments
                selected = data["selected"]
                visible = selected.get('is_manual', False)
                
            elif filter_type == "Show Unselected":
                # Show segments that haven't been reviewed
                visible = not data.get("reviewed", False)
                
            item.setHidden(not visible)
            
    def show_context_menu(self, position):
        """Show enhanced context menu"""
        item = self.main_content.itemAt(position)
        if not item:
            return
            
        menu = QMenu(self)
        
        # Model selection options
        data = item.data(0, Qt.UserRole)
        if data:
            segments = data["segments"]
            
            model_menu = menu.addMenu("Select Model")
            for segment in segments:
                action = model_menu.addAction(f"{segment['model']}: \"{segment['text'][:50]}...\"")
                action.triggered.connect(lambda checked, seg=segment: self.select_model_for_item(item, seg))
                
        menu.addSeparator()
        
        # Editing options
        edit_action = menu.addAction("Edit Text (F2)")
        edit_action.triggered.connect(lambda: self.main_content.editItem(item, 4))
        
        # Audio options
        if self.audio_player:
            play_action = menu.addAction("Play Segment Audio (Space)")
            play_action.triggered.connect(self.play_segment_audio)
            
        menu.addSeparator()
        
        # Delete option
        delete_action = menu.addAction("Delete Segment (Del)")
        delete_action.triggered.connect(lambda: self.delete_segment(item))
        
        menu.exec_(self.main_content.mapToGlobal(position))

    def select_model_for_item(self, item, segment, clicked_column):
        """
        Select model with enhanced text merging and clean visual feedback
        
        This method now:
        1. Intelligently merges similar texts with optimal punctuation
        2. Preserves internal punctuation while prioritizing ending punctuation
        3. Provides clean visual feedback without text markers
        4. Handles manual edits and selections properly
        """
        
        # Save state for undo functionality
        self.save_state_for_undo(f"Select {segment['model']}")
        
        # Get item data
        data = item.data(0, Qt.UserRole)
        if not data:
            self.log("Error: No data found for item", error=True)
            return
        
        # Get relevant data
        edited_texts = data.get("edited_texts", {})
        selected_model = segment["model"]
        model_segments = data.get("model_segments", {})
        clip_idx = data["clip_idx"]
        group_idx = data["group_idx"]
        
        self.log(f"Selecting model {selected_model} for clip {clip_idx}, group {group_idx}")
        
        # ENHANCED: Intelligent text processing with merging
        final_text = segment.get("text", "").strip()
        text_enhancement_applied = False
        
        # Check if this segment group has multiple models with similar content
        if len(model_segments) > 1:
            # Create segment group for analysis
            segment_group = []
            for model_name, model_seg in model_segments.items():
                segment_group.append({
                    'text': model_seg.get('text', '').strip(),
                    'model': model_name,
                    'start': model_seg.get('start', 0),
                    'end': model_seg.get('end', 0)
                })
            
            # Check if segments have meaningful differences
            if not TextComparisonUtils.segment_group_has_differences(segment_group):
                # Segments are essentially the same - use intelligent merging
                original_text = final_text
                merged_text = TextComparisonUtils.get_best_merged_text_from_group(segment_group)
                
                if merged_text and merged_text != original_text:
                    final_text = merged_text
                    text_enhancement_applied = True
                    
                    self.log(f"Enhanced text merging applied:")
                    self.log(f"  Original: '{original_text}'")
                    self.log(f"  Enhanced: '{final_text}'")
                    
                    # Log details about the enhancement
                    original_ending = TextComparisonUtils._extract_ending_punctuation(original_text)
                    enhanced_ending = TextComparisonUtils._extract_ending_punctuation(final_text)
                    
                    if original_ending != enhanced_ending:
                        self.log(f"  Ending punctuation: '{original_ending}' → '{enhanced_ending}'")
                    
                    internal_punct_original = TextComparisonUtils._count_internal_punctuation(original_text)
                    internal_punct = TextComparisonUtils._count_internal_punctuation(final_text)
                    
                    if internal_punct > internal_punct_original:
                        self.log(f"  Internal punctuation preserved: {internal_punct} marks")
            else:
                # Segments have meaningful differences - use selected text as-is
                self.log(f"Segments have differences - using selected text as-is")
        
        # Handle edited text override
        if selected_model in edited_texts:
            # User has manually edited this model's text - use the edited version
            edited_text = edited_texts[selected_model].strip()
            self.log(f"Using manually edited text: '{edited_text}'")
            final_text = edited_text
            is_edited = True
            original_text = segment.get("text", "")
        else:
            # Use the final processed text (enhanced or original)
            is_edited = False
            original_text = segment.get("text", "") if text_enhancement_applied else ""
        
        # Create the final segment with all enhancements
        final_segment = segment.copy()
        final_segment["text"] = final_text
        final_segment["is_edited"] = is_edited
        final_segment["is_enhanced"] = text_enhancement_applied
        final_segment["original_text"] = original_text if (is_edited or text_enhancement_applied) else ""
        
        # Store enhancement details for potential display
        if text_enhancement_applied:
            final_segment["enhancement_details"] = {
                "merged_from_models": [seg['model'] for seg in segment_group],
                "original_text": segment.get("text", ""),
                "enhanced_text": final_text
            }
        
        # Update selection data
        data["selected"] = final_segment
        item.setData(0, Qt.UserRole, data)
        
        # Update global selected transcriptions
        if not hasattr(self, 'selected_transcriptions'):
            self.selected_transcriptions = {}
        
        if clip_idx not in self.selected_transcriptions:
            self.selected_transcriptions[clip_idx] = {}
        
        self.selected_transcriptions[clip_idx][group_idx] = final_segment
        
        # ENHANCED VISUAL FEEDBACK - Clean and informative
        # Reset all column styling first
        for i, model in enumerate(self.model_columns):
            col_index = 3 + i
            
            # Reset to default styling
            item.setBackground(col_index, QColor(255, 255, 255))  # White background
            font = item.font(col_index)
            font.setBold(False)
            font.setItalic(False)
            item.setFont(col_index, font)
            item.setForeground(col_index, QColor(0, 0, 0))  # Black text
            
            # Update display text for each model
            if model in model_segments:
                original_segment = model_segments[model]
                display_text = original_segment.get("text", "")
                
                # Show edited text if available (clean, no markers)
                if model in edited_texts:
                    display_text = edited_texts[model]
                    
                    # Style edited text columns
                    item.setBackground(col_index, QColor(230, 255, 230))  # Light green
                    font = item.font(col_index)
                    font.setItalic(True)
                    font.setBold(True)
                    item.setFont(col_index, font)
                    item.setForeground(col_index, QColor(0, 120, 0))  # Dark green
                
                # Update the text display
                item.setText(col_index, display_text)
        
        # Highlight the selected column with appropriate styling
        selected_col_index = clicked_column
        
        if is_edited:
            # Selected model with manual edits
            item.setBackground(selected_col_index, QColor(180, 255, 180))  # Darker green
            font = item.font(selected_col_index)
            font.setBold(True)
            font.setItalic(True)
            item.setFont(selected_col_index, font)
            item.setForeground(selected_col_index, QColor(0, 100, 0))  # Dark green
            
            self.log(f"✓ Selected {selected_model} with manual edits")
            
        elif text_enhancement_applied:
            # Selected model with text enhancement
            item.setBackground(selected_col_index, QColor(200, 230, 255))  # Light blue
            font = item.font(selected_col_index)
            font.setBold(True)
            item.setFont(selected_col_index, font)
            item.setForeground(selected_col_index, QColor(0, 60, 120))  # Dark blue
            
            # Update the display to show enhanced text
            item.setText(selected_col_index, final_text)
            
            self.log(f"✓ Selected {selected_model} with text enhancement")
            
        else:
            # Regular selection
            item.setBackground(selected_col_index, QColor(200, 255, 200))  # Light green
            font = item.font(selected_col_index)
            font.setBold(True)
            item.setFont(selected_col_index, font)
            
            self.log(f"✓ Selected {selected_model}")
        
        # Add visual indicator for enhancement in the selected column
        if text_enhancement_applied and not is_edited:
            # Add a subtle indicator that text was enhanced
            current_text = item.text(selected_col_index)
            # We keep the text clean but the blue background indicates enhancement
            
            # Optional: Add tooltip with enhancement details
            tooltip_text = f"Enhanced text from {len(segment_group)} models\n"
            tooltip_text += f"Original: '{segment.get('text', '')}'\n"
            tooltip_text += f"Enhanced: '{final_text}'"
            
            # Note: QTreeWidget doesn't support per-cell tooltips easily,
            # but we can store this info for potential display elsewhere
            item.setToolTip(selected_col_index, tooltip_text)
        
        # Update status display to reflect changes
        self.update_status_display()
        
        # Log summary of action taken
        action_summary = []
        if is_edited:
            action_summary.append("manual edit")
        if text_enhancement_applied:
            action_summary.append("text enhancement")
        
        if action_summary:
            self.log(f"Selection complete with: {', '.join(action_summary)}")
        else:
            self.log(f"Selection complete")
        
        # Optional: Auto-save progress after significant changes
        if is_edited or text_enhancement_applied:
            # Could implement auto-save here
            pass

    def play_segment_audio(self):
        """Play audio for the current segment with padding offset"""
        current_item = self.main_content.currentItem()
        if not current_item or not self.audio_player:
            return
            
        data = current_item.data(0, Qt.UserRole)
        if not data:
            return
            
        segments = data.get("segments", [])
        if segments:
            # Get time range for this segment
            start_time = min(seg['start'] for seg in segments)
            end_time = max(seg['end'] for seg in segments)
            
            # Start playback at segment start (after padding)
            self.audio_player.media_player.set_time(int(start_time * 1000))
            self.audio_player.start_playback()
            
            # Store for segment timer
            self.current_segment_start = start_time
            self.current_segment_end = end_time
            
            # Start segment timer if available
            if hasattr(self, 'segment_timer'):
                self.segment_timer.start()

    def on_audio_stopped(self):
        """Handle audio playback stopped"""
        # Update UI state when audio stops
        pass
        
    def update_segment_styling(self, item, selected_segment):
        """Update visual styling for a segment item"""
        if selected_segment.get('is_manual', False):
            # Manual edit styling
            item.setBackground(3, QColor(220, 255, 220))
            font = item.font(3)
            font.setBold(True)
            font.setItalic(True)
            item.setFont(3, font)
            
        elif selected_segment.get('is_base', False):
            # Base model selected
            item.setBackground(3, QColor(200, 230, 255))
            font = item.font(3)
            font.setBold(True)
            item.setFont(3, font)
            
        else:
            # Comparison model selected
            item.setBackground(3, QColor(200, 255, 200))
            font = item.font(3)
            font.setBold(True)
            item.setFont(3, font)
            
    def update_status_display(self):
        """Update the status bar with current statistics including edited text count"""
        total_segments = self.main_content.topLevelItemCount()
        visible_segments = sum(1 for i in range(total_segments) 
                            if not self.main_content.topLevelItem(i).isHidden())
        selected_count = len(self.main_content.selectedItems())
        
        # Count manual edits and edited texts
        manual_edits = 0
        edited_texts = 0
        for clip_selections in self.selected_transcriptions.values():
            for selection in clip_selections.values():
                if selection.get('is_manual', False):
                    manual_edits += 1
                if selection.get('is_edited', False):
                    edited_texts += 1
        
        # Count total edited texts in all items
        total_edited_texts = 0
        for i in range(total_segments):
            item = self.main_content.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            if data and "edited_texts" in data:
                total_edited_texts += len(data["edited_texts"])
        
        self.selected_count_label.setText(f"{selected_count} selected")
        self.total_segments_label.setText(f"{visible_segments}/{total_segments} segments")
        
        # Update manual edits label to include edited texts
        edits_text = f"{manual_edits} manual edits"
        if total_edited_texts > 0:
            edits_text += f", {total_edited_texts} text edits"
        self.manual_edits_label.setText(edits_text)

    def auto_select_best(self):
        """Auto-select best transcriptions using enhanced heuristics"""
        self.save_state_for_undo("Auto-select best transcriptions")
        
        improved_count = 0
        
        for i in range(self.main_content.topLevelItemCount()):
            item = self.main_content.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            
            if not data or data.get("deleted", False):
                continue
            
            model_segments = data.get("model_segments", {})
            if len(model_segments) <= 1:
                continue
            
            # Enhanced best selection logic
            best_segment = self.select_best_segment(list(model_segments.values()))
            current_selection = data["selected"]
            
            if best_segment != current_selection:
                # Find the column for this model
                best_model = best_segment["model"]
                if best_model in self.model_columns:
                    model_index = self.model_columns.index(best_model)
                    clicked_column = 3 + model_index
                    self.select_model_for_item(item, best_segment, clicked_column)
                    improved_count += 1
        
        QMessageBox.information(
            self, 
            "Auto-Selection Complete", 
            f"Improved {improved_count} transcriptions using enhanced AI selection."
        )

    def select_best_segment(self, segments):
        """Enhanced best segment selection with length preference for similar texts"""
        if not segments:
            return None
            
        # Score each segment
        scored_segments = []
        
        # First, group segments by similarity
        similar_groups = []
        remaining_segments = segments.copy()
        
        while remaining_segments:
            current_segment = remaining_segments.pop(0)
            current_group = [current_segment]
            
            # Find similar segments
            i = 0
            while i < len(remaining_segments):
                if TextComparisonUtils.are_texts_essentially_same(
                    current_segment.get('text', ''), 
                    remaining_segments[i].get('text', '')
                ):
                    current_group.append(remaining_segments.pop(i))
                else:
                    i += 1
            
            similar_groups.append(current_group)
        
        # For each group, if they're similar, prefer the longest
        for group in similar_groups:
            if len(group) > 1:
                # Multiple similar segments - choose the longest
                longest_segment = max(group, key=lambda x: len(x.get('text', '')))
                scored_segments.append((1000 + len(longest_segment.get('text', '')), longest_segment))
                print(f"📏 Chose longest similar text: '{longest_segment.get('text', '')[:50]}...'")
            else:
                # Single segment - score normally
                segment = group[0]
                score = 0
                text = segment.get('text', '')
                
                # Length bonus (longer is often more complete)
                score += len(text) * 0.1
                
                # Capitalization bonus
                if text and text[0].isupper():
                    score += 5
                    
                # Punctuation bonus
                if text.endswith('.') or text.endswith('!') or text.endswith('?'):
                    score += 3
                    
                # Grammar quality (simple heuristics)
                if ' ' in text:  # Multi-word
                    score += 2
                if any(word in text.lower() for word in ['the', 'and', 'a', 'to', 'of']):
                    score += 1  # Common English words
                    
                # Model preference (could be configured)
                model_bonuses = {
                    'large-v3': 2,
                    'large-v2': 1,
                    'turbo': 1
                }
                score += model_bonuses.get(segment.get('model', ''), 0)
                
                scored_segments.append((score, segment))
        
        # Return highest scoring segment
        return max(scored_segments, key=lambda x: x[0])[1]

    def select_all_base_model(self):
        """Select base model for all segments"""
        if not hasattr(self, 'model_columns'):
            return
        
        self.save_state_for_undo("Select all base model")
        
        changed_count = 0
        
        for i in range(self.main_content.topLevelItemCount()):
            item = self.main_content.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            
            if not data or data.get("deleted", False):
                continue
            
            segments = data.get("segments", [])
            base_segment = next((seg for seg in segments if seg.get('is_base', False)), None)
            
            if base_segment and base_segment != data["selected"]:
                # Find the column for the base model
                base_model = base_segment["model"]
                if base_model in self.model_columns:
                    model_index = self.model_columns.index(base_model)
                    clicked_column = 3 + model_index
                    self.select_model_for_item(item, base_segment, clicked_column)
                    changed_count += 1
        
        QMessageBox.information(
            self, 
            "Base Model Selected", 
            f"Selected base model for {changed_count} segments."
        )

    def export_results(self):
        """Export detailed results including edit history"""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Enhanced Results", 
            "enhanced_transcription_results.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("ENHANCED TRANSCRIPTION EDITING RESULTS\n")
                    f.write("=" * 60 + "\n\n")
                    
                    # Export editing statistics
                    f.write("EDITING STATISTICS\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Total segments: {self.main_content.topLevelItemCount()}\n")
                    f.write(f"Undo operations performed: {len(self.undo_stack)}\n")
                    
                    manual_edits = sum(1 for clip_selections in self.selected_transcriptions.values()
                                     for selection in clip_selections.values() 
                                     if selection.get('is_manual', False))
                    f.write(f"Manual edits: {manual_edits}\n\n")
                    
                    # Export final transcriptions
                    f.write("FINAL TRANSCRIPTIONS\n")
                    f.write("-" * 20 + "\n")
                    
                    for i, result in enumerate(self.comparison_results):
                        f.write(f"CLIP {i+1}: {result.clip_info.get('name', 'Unknown')}\n")
                        
                        clip_selections = self.selected_transcriptions.get(i, {})
                        for group_idx in sorted(clip_selections.keys()):
                            segment = clip_selections[group_idx]
                            f.write(f"  {segment['start']:.1f}s: [{segment['model']}] {segment['text']}\n")
                        
                        f.write("\n")
                
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")

    def create_final_srt_files(self):
        """Create final SRT files with all enhancements"""
        if not self.selected_transcriptions:
            QMessageBox.warning(self, "No Selections", "No transcriptions have been selected.")
            return
        
        # Show progress dialog for SRT creation
        from PyQt5.QtWidgets import QProgressDialog
        
        progress = QProgressDialog("Creating enhanced SRT files...", "Cancel", 0, len(self.comparison_results), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            output_dir = self.parent().output_dir if self.parent() else os.getcwd()
            created_files = []
            
            for clip_idx, result in enumerate(self.comparison_results):
                if progress.wasCanceled():
                    break
                
                progress.setValue(clip_idx)
                progress.setLabelText(f"Processing {result.clip_info.get('name', f'Clip {clip_idx+1}')}...")
                QApplication.processEvents()
                
                # Get selected transcriptions for this clip
                clip_selections = self.selected_transcriptions.get(clip_idx, {})
                if not clip_selections:
                    continue
                
                # Create segments from selected transcriptions
                selected_segments = []
                for group_idx in sorted(clip_selections.keys()):
                    selected_segment = clip_selections[group_idx]
                    
                    # Convert to segment-like object
                    class SelectedSegment:
                        def __init__(self, start, end, text):
                            self.start = start
                            self.end = end
                            self.text = text
                    
                    seg = SelectedSegment(
                        selected_segment.get('start', 0),
                        selected_segment.get('end', 0),
                        selected_segment.get('text', '')
                    )
                    selected_segments.append(seg)
                
                if not selected_segments:
                    continue
                
                # Create SRT file
                clip_name = result.clip_info.get("name", f"Clip_{clip_idx+1}")
                safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in clip_name)
                srt_path = os.path.join(output_dir, f"{safe_name}_comparison_selected.srt")
                
                success = self.create_srt_from_segments(selected_segments, srt_path)
                if success:
                    created_files.append(srt_path)
            
            progress.setValue(len(self.comparison_results))
            
            if created_files:
                # FIXED: Show concise success message instead of listing all files
                message = f"✅ Successfully created {len(created_files)} SRT files!\n\n"
                message += f"📁 Location: {output_dir}\n\n"
                
                # Show first few files as examples if there are many
                if len(created_files) <= 5:
                    message += "Files created:\n"
                    for file_path in created_files:
                        message += f"• {os.path.basename(file_path)}\n"
                else:
                    message += "Sample files created:\n"
                    for file_path in created_files[:3]:
                        message += f"• {os.path.basename(file_path)}\n"
                    message += f"• ... and {len(created_files) - 3} more files\n"
                
                message += f"\n💡 These contain your manually selected best transcriptions from model comparison."
                
                QMessageBox.information(self, "SRT Files Created", message)
            else:
                QMessageBox.warning(self, "No Files Created", "No SRT files were created. Please make sure you have made selections.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error Creating SRT Files", f"Error: {str(e)}")
        finally:
            progress.close()

    def create_srt_from_segments(self, segments, srt_path):
        """Create SRT file from selected segments"""
        try:
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, start=1):
                    if not segment.text or not segment.text.strip():
                        continue
                    
                    # Write segment number
                    f.write(f"{i}\n")
                    
                    # Write timestamps
                    start_time = self.format_time_for_srt(segment.start)
                    end_time = self.format_time_for_srt(segment.end)
                    f.write(f"{start_time} --> {end_time}\n")
                    
                    # Write text
                    f.write(f"{segment.text.strip()}\n\n")
            
            return True
        except Exception as e:
            print(f"Error creating SRT file: {str(e)}")
            return False

    def format_time_for_srt(self, seconds):
        """Format seconds to SRT time format"""
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def combine_all_segments(self, result):
        """Combine segments from all models (from original code)"""
        all_segments = []
        
        # Add base model segments
        for seg in result.base_segments:
            all_segments.append({
                'model': result.base_model,
                'start': seg.start,
                'end': seg.end,
                'text': seg.text.strip(),
                'is_base': True
            })
        
        # Add comparison model segments
        for comp_result in result.comparison_results:
            for seg in comp_result["segments"]:
                all_segments.append({
                    'model': comp_result["model"],
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text.strip(),
                    'is_base': False
                })
        
        return all_segments

    def get_selected_transcriptions_for_combined_srt(self):
        """Get selected transcriptions formatted for combined SRT creation - FIXED to ensure edited text is included"""
        selected_data = []
        
        if not hasattr(self, 'selected_transcriptions'):
            return selected_data
        
        for clip_idx, result in enumerate(self.comparison_results):
            clip_info = result.clip_info
            
            # Get selected transcriptions for this clip
            clip_selections = self.selected_transcriptions.get(clip_idx, {})
            if not clip_selections:
                continue
            
            # Create segments from selected transcriptions
            selected_segments = []
            for group_idx in sorted(clip_selections.keys()):
                selected_segment = clip_selections[group_idx]
                
                # FIXED: Ensure we use the edited text and clean it
                text_to_use = selected_segment.get('text', '').strip()
                is_edited = selected_segment.get('is_edited', False)
                
                # Debug logging to verify edited text is being used
                if is_edited:
                    self.log(f"Using edited text for clip {clip_idx}, group {group_idx}: '{text_to_use}'")
                
                # Create a segment-like object with the selected/edited text
                class SelectedSegment:
                    def __init__(self, start, end, text, is_edited=False):
                        self.start = start
                        self.end = end
                        self.text = text
                        self.is_edited = is_edited
                
                seg = SelectedSegment(
                    selected_segment.get('start', 0),
                    selected_segment.get('end', 0),
                    text_to_use,  # This should now include edited text
                    is_edited
                )
                selected_segments.append(seg)
            
            if selected_segments:
                # Create clip data structure expected by the SRT creation functions
                clip_data = {
                    "clip_idx": clip_idx,
                    "name": clip_info.get("name", f"Clip {clip_idx + 1}"),
                    "track_index": clip_info.get("track_index", 0),
                    "timeline_start": clip_info.get("timeline_start", 0),
                    "timeline_end": clip_info.get("timeline_end", 0),
                    "start_time": clip_info.get("start_time", 0),
                    "duration": clip_info.get("duration", 0),
                    "segments": selected_segments,
                    "file_path": clip_info.get("file_path", ""),
                    "output_path": clip_info.get("output_path", "")
                }
                selected_data.append(clip_data)
        
        self.log(f"Prepared {len(selected_data)} clips with selected transcriptions for SRT creation")
        return selected_data

    def play_specific_segment(self, segment):
        """Play audio for a specific segment with correct 1-second padding offset"""
        if not self.audio_player:
            return
        
        try:
            # Stop current playback
            self.audio_player.pause_playback()

            start_time = segment['start']
            end_time = segment['end']
            
            # Start playback
            self.audio_player.start_playback()
            
            # Seek to segment start (after padding)
            self.audio_player.media_player.set_time(int(start_time * 1000))
            
            # Store segment info for potential looping
            self.current_segment_start = start_time
            self.current_segment_end = end_time
            self.playing_segment = segment
            
            # Start segment timer for precise control
            if hasattr(self, 'segment_timer'):
                self.segment_timer.start()

            # Update status display
            model_name = segment.get('model', 'Unknown')
            if hasattr(self, 'audio_player') and hasattr(self.audio_player, 'status_label'):
                self.audio_player.status_label.setText(
                    f"Playing {model_name} segment ({segment['start']:.1f}s-{segment['end']:.1f}s) [+1s padding offset]"
                )
            
        except Exception as e:
            if self.audio_player and hasattr(self.audio_player, 'status_label'):
                self.audio_player.status_label.setText(f"Segment playback error: {str(e)}")

    def log(self, message):
        print(f"[ModelComparison] {message}")

class SegmentSelectionDialog(QDialog):
    """Enhanced dialog for selecting between different model transcriptions with audio playback and manual editing"""
    
    def __init__(self, segments, current_selection, audio_file_path=None, parent=None):
        super().__init__(parent)
        self.segments = segments
        self.current_selection = current_selection
        self.selected_segment = current_selection
        self.audio_file_path = audio_file_path
        self.manual_edit_mode = False
        self.custom_text = ""
        
        # Initialize audio player as None first
        self.audio_player = None
        self.setup_segment_timer()
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the enhanced selection dialog UI"""
        self.setWindowTitle("Select Transcription with Audio Preview & Manual Editor")
        self.setMinimumSize(900, 700)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Header with time range info
        start_time = min(seg['start'] for seg in self.segments)
        end_time = max(seg['end'] for seg in self.segments)
        header = QLabel(f"Select transcription for time range {start_time:.1f}s - {end_time:.1f}s")
        header.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(header)
        
        # Audio player section - with proper error handling
        if self.audio_file_path and os.path.exists(self.audio_file_path):
            try:
                audio_group = QFrame()
                audio_group.setFrameStyle(QFrame.Box)
                audio_group.setStyleSheet("QFrame { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; }")
                audio_layout = QVBoxLayout(audio_group)
                
                audio_label = QLabel("🎵 Audio Preview")
                audio_label.setFont(QFont("Arial", 10, QFont.Bold))
                audio_layout.addWidget(audio_label)
                
                # Create audio player with error handling
                self.audio_player = AudioPlayerWidget()
                self.audio_player.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
                audio_layout.addWidget(self.audio_player)
                
                # Load the audio file
                if self.audio_player.load_audio_file(self.audio_file_path):
                    # Add segment-specific controls
                    segment_controls_layout = QHBoxLayout()
                    
                    self.play_segment_button = QPushButton("▶ Play This Segment")
                    self.play_segment_button.setStyleSheet("""
                        QPushButton {
                            background-color: #007bff;
                            color: white;
                            border: none;
                            padding: 8px 16px;
                            border-radius: 4px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background-color: #0056b3;
                        }
                        QPushButton:pressed {
                            background-color: #004085;
                        }
                    """)
                    self.play_segment_button.clicked.connect(self.play_current_segment)
                    
                    self.loop_segment_button = QPushButton("🔄 Loop Off")
                    self.loop_segment_button.setCheckable(True)
                    self.loop_segment_button.setStyleSheet("""
                        QPushButton {
                            background-color: #6c757d;
                            color: white;
                            border: none;
                            padding: 8px 16px;
                            border-radius: 4px;
                        }
                        QPushButton:checked {
                            background-color: #28a745;
                        }
                        QPushButton:hover {
                            background-color: #545b62;
                        }
                        QPushButton:checked:hover {
                            background-color: #218838;
                        }
                    """)
                    self.loop_segment_button.clicked.connect(self.toggle_loop_segment)
                    self.loop_enabled = False
                    
                    segment_controls_layout.addWidget(self.play_segment_button)
                    segment_controls_layout.addWidget(self.loop_segment_button)
                    segment_controls_layout.addStretch()
                    
                    audio_layout.addLayout(segment_controls_layout)
                    
                    # Connect to player events for looping
                    self.audio_player.playback_stopped.connect(self.on_playback_stopped)
                    
                else:
                    # Audio loading failed
                    self.audio_player = None
                    error_label = QLabel("⚠ Failed to load audio file")
                    error_label.setStyleSheet("color: #dc3545; padding: 10px;")

                layout.addWidget(audio_group)
                
            except Exception as e:
                # Entire audio section failed
                self.audio_player = None
                error_label = QLabel(f"⚠ Audio preview unavailable: {str(e)}")
                error_label.setStyleSheet("color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 5px;")
                layout.addWidget(error_label)
        else:
            # No audio available
            self.audio_player = None
            no_audio_label = QLabel("⚠ Audio file not available for preview")
            no_audio_label.setStyleSheet("color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 5px;")
            layout.addWidget(no_audio_label)
        
        # Instructions
        instructions = QLabel(
            "Click on any transcription option to select it, or use the manual editor to create your own transcription. "
            + ("Use the audio player above to listen and compare the options." if self.audio_player else "")
        )
        instructions.setStyleSheet("color: #666; font-style: italic; margin: 10px 0;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Create tab widget for AI transcriptions vs Manual editor
        self.tab_widget = QTabWidget()
        
        # Tab 1: AI Transcriptions
        ai_tab = QWidget()
        ai_layout = QVBoxLayout(ai_tab)
        
        # Radio button group for transcription options
        self.button_group = QButtonGroup(self)
        self.clickable_frames = []
        
        for i, segment in enumerate(self.segments):
            # Create clickable frame for each option
            frame = ClickableFrame()
            frame.setFrameStyle(QFrame.Box)
            frame.setCursor(Qt.PointingHandCursor)
            frame_layout = QHBoxLayout(frame)
            
            # Radio button
            radio = QRadioButton()
            self.button_group.addButton(radio, i)
            
            # Model info section
            model_info_layout = QVBoxLayout()
            
            # Model name
            model_label = QLabel(f"{segment['model']}")
            model_label.setMinimumWidth(120)
            model_label.setFont(QFont("Arial", 10, QFont.Bold))
            
            # Time info
            time_label = QLabel(f"{segment['start']:.1f}s - {segment['end']:.1f}s")
            time_label.setStyleSheet("color: #666; font-size: 9px;")
            
            model_info_layout.addWidget(model_label)
            model_info_layout.addWidget(time_label)
            
            # Text content
            text_label = QLabel(f'"{segment["text"]}"')
            text_label.setWordWrap(True)
            text_label.setStyleSheet("padding: 10px; font-size: 11px;")
            
            # Set styling based on model type
            if segment['is_base']:
                model_label.setStyleSheet("color: #2196F3; font-weight: bold;")
                frame.setStyleSheet("""
                    ClickableFrame { 
                        background-color: #f0f8ff; 
                        border: 2px solid #ccc;
                        border-radius: 5px;
                        margin: 2px;
                    }
                    ClickableFrame:hover { 
                        background-color: #e0f0ff; 
                        border: 2px solid #2196F3;
                    }
                """)
            else:
                model_label.setStyleSheet("color: #FF9800; font-weight: bold;")
                frame.setStyleSheet("""
                    ClickableFrame { 
                        background-color: #fff8f0; 
                        border: 2px solid #ccc;
                        border-radius: 5px;
                        margin: 2px;
                    }
                    ClickableFrame:hover { 
                        background-color: #ffe8d0; 
                        border: 2px solid #FF9800;
                    }
                """)
            
            # Add individual play button for this segment (only if audio player exists)
            if self.audio_player:
                play_this_button = QPushButton("▶")
                play_this_button.setFixedSize(30, 30)
                play_this_button.setToolTip(f"Play {segment['model']} segment")
                play_this_button.setStyleSheet("""
                    QPushButton {
                        background-color: #28a745;
                        color: white;
                        border: none;
                        border-radius: 15px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #218838;
                    }
                """)
                play_this_button.clicked.connect(lambda checked, seg=segment: self.play_specific_segment(seg))
                frame_layout.addWidget(play_this_button)
            
            frame_layout.addWidget(radio)
            frame_layout.addLayout(model_info_layout)
            frame_layout.addWidget(text_label, 1)
            
            ai_layout.addWidget(frame)
            
            # Store references
            frame.radio_button = radio
            frame.segment_index = i
            frame.segment_data = segment
            
            # Connect frame click to radio button selection
            frame.clicked.connect(lambda idx=i: self.select_option(idx))
            
            self.clickable_frames.append(frame)
            
            # Select current selection
            if segment == self.current_selection:
                radio.setChecked(True)
                self.update_frame_selection(i, True)
        
        # Connect radio button signal
        self.button_group.buttonClicked.connect(self.on_radio_selection_changed)
        
        self.tab_widget.addTab(ai_tab, "AI Transcriptions")
        
        # Tab 2: Manual Editor
        manual_tab = QWidget()
        manual_layout = QVBoxLayout(manual_tab)
        
        # Manual editor header
        manual_header = QLabel("✏️ Manual Transcription Editor")
        manual_header.setFont(QFont("Arial", 12, QFont.Bold))
        manual_layout.addWidget(manual_header)
        
        # Editor instructions
        editor_instructions = QLabel(
            "Use this editor to manually create or edit the transcription. "
            + ("Listen to the audio and type exactly what you hear." if self.audio_player else "Type exactly what you hear.")
        )
        editor_instructions.setStyleSheet("color: #666; font-style: italic; margin: 5px 0;")
        editor_instructions.setWordWrap(True)
        manual_layout.addWidget(editor_instructions)
        
        # Text editor
        self.manual_text_edit = QTextEdit()
        self.manual_text_edit.setPlaceholderText("Type your manual transcription here...")
        self.manual_text_edit.setMinimumHeight(200)
        self.manual_text_edit.setStyleSheet("""
            QTextEdit {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
                font-family: Arial, sans-serif;
                background-color: white;
            }
            QTextEdit:focus {
                border-color: #007bff;
            }
        """)
        
        # Load existing text if we have it
        if hasattr(self, 'custom_text') and self.custom_text:
            self.manual_text_edit.setPlainText(self.custom_text)
        elif self.current_selection:
            # Pre-populate with current selection for editing
            self.manual_text_edit.setPlainText(self.current_selection['text'])
        
        manual_layout.addWidget(self.manual_text_edit)
        
        # Editor controls
        editor_controls_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Clear Text")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.clear_button.clicked.connect(self.clear_manual_text)
        
        self.copy_ai_button = QPushButton("Copy from AI Selection")
        self.copy_ai_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        self.copy_ai_button.clicked.connect(self.copy_from_ai_selection)
        
        # Word count display
        self.word_count_label = QLabel("Words: 0")
        self.word_count_label.setStyleSheet("color: #666; font-size: 10px;")
        
        # Connect text change to update word count
        self.manual_text_edit.textChanged.connect(self.update_word_count)
        
        editor_controls_layout.addWidget(self.clear_button)
        editor_controls_layout.addWidget(self.copy_ai_button)
        editor_controls_layout.addStretch()
        editor_controls_layout.addWidget(self.word_count_label)
        
        manual_layout.addLayout(editor_controls_layout)
        
        # Add manual editor tab
        self.tab_widget.addTab(manual_tab, "Manual Editor")
        
        # Connect tab change
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        layout.addWidget(self.tab_widget)
        
        # Selection status
        self.selection_status_label = QLabel()
        self.selection_status_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #f8f9fa; border-radius: 3px;")
        self.update_selection_status()
        layout.addWidget(self.selection_status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Play controls info (only if audio player exists)
        if self.audio_player:
            info_label = QLabel("💡 Tip: Use individual ▶ buttons to compare each transcription's audio segment")
            info_label.setStyleSheet("color: #17a2b8; font-size: 10px; font-style: italic;")
            button_layout.addWidget(info_label)
        
        button_layout.addStretch()
        
        self.ok_button = QPushButton("Use This Transcription")
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #545b62;
            }
        """)
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        # Initialize word count
        self.update_word_count()

    def setup_segment_timer(self):
        """Set up segment timer for precise playback control - call this in __init__"""
        self.segment_timer = QTimer()
        self.segment_timer.timeout.connect(self.check_segment_end)
        self.segment_timer.setInterval(100)  # Check every 100ms
        
        # Initialize segment tracking variables
        self.current_segment_start = 0
        self.current_segment_end = 0
        self.playing_segment = None

    def check_segment_end(self):
        """Check if we've reached the end of the current segment"""
        if not self.audio_player or not hasattr(self, 'current_segment_end'):
            return
        
        try:
            # Get current playback position in seconds
            current_time_ms = self.audio_player.media_player.get_time()
            if current_time_ms < 0:  # VLC returns -1 if no media
                return
                
            current_time = current_time_ms / 1000.0
            
            # Check if we've passed the segment end
            if current_time >= self.current_segment_end:
                self.segment_timer.stop()
                
                # Check if looping is enabled
                if hasattr(self, 'loop_enabled') and self.loop_enabled and hasattr(self, 'playing_segment'):
                    # Restart the segment
                    self.play_specific_segment(self.playing_segment)
                else:
                    # Pause at segment end
                    self.audio_player.pause_playback()
                    if hasattr(self.audio_player, 'status_label'):
                        self.audio_player.status_label.setText("Segment finished")
                    
        except Exception as e:
            # Silently handle errors to avoid log spam
            pass

    def play_current_segment(self):
        if not self.audio_player:
            return
        
        if self.manual_edit_mode or not self.selected_segment:
            # In manual mode or no AI selection, play the general segment
            if self.segments:
                self.play_specific_segment(self.segments[0])
        else:
            # Play the selected AI segment
            self.play_specific_segment(self.selected_segment)

    def play_specific_segment(self, segment):
        """Play audio for a specific segment with correct 1-second padding offset"""
        if not self.audio_player:
            return
        
        try:
            # Stop current playback
            self.audio_player.pause_playback()

            start_time = segment['start']
            end_time = segment['end']
            
            # Start playback
            self.audio_player.start_playback()
            
            # Seek to segment start
            self.audio_player.media_player.set_time(int(start_time * 1000))
            
            # Store segment info for potential looping
            self.current_segment_start = start_time
            self.current_segment_end = end_time
            self.playing_segment = segment
            
            # Start segment timer for precise control
            if hasattr(self, 'segment_timer'):
                self.segment_timer.start()

            # Update status display
            model_name = segment.get('model', 'Unknown')
            if hasattr(self, 'audio_player') and hasattr(self.audio_player, 'status_label'):
                self.audio_player.status_label.setText(
                    f"Playing {model_name} segment ({segment['start']:.1f}s-{segment['end']:.1f}s)"
                )
            
        except Exception as e:
            if self.audio_player and hasattr(self.audio_player, 'status_label'):
                self.audio_player.status_label.setText(f"Segment playback error: {str(e)}")

    def stop_segment_playback(self):
        """Stop playback at the end of the segment"""
        if self.audio_player and self.audio_player.is_playing:
            # Check if we should loop
            if hasattr(self, 'loop_enabled') and self.loop_enabled and hasattr(self, 'playing_segment'):
                # Restart the segment
                self.play_specific_segment(self.playing_segment)
            else:
                # Just pause (don't stop completely so user can continue if they want)
                self.audio_player.pause_playback()
    
    def toggle_loop_segment(self):
        """Toggle segment looping"""
        if hasattr(self, 'loop_segment_button'):
            self.loop_enabled = self.loop_segment_button.isChecked()
            if self.loop_enabled:
                self.loop_segment_button.setText("🔄 Loop On")
            else:
                self.loop_segment_button.setText("🔄 Loop Off")
    
    def on_playback_stopped(self):
        """Handle when playback stops naturally"""
        # Clear segment tracking
        if hasattr(self, 'playing_segment'):
            delattr(self, 'playing_segment')
    
    def select_option(self, index):
        """Select an option by index"""
        # Update radio button
        radio = self.button_group.button(index)
        if radio:
            radio.setChecked(True)
        
        # Update selection
        self.selected_segment = self.segments[index]
        
        # Update visual feedback
        self.update_all_frame_selections()
        
        # Update status
        self.update_selection_status()
        
        # Auto-update manual editor if it's currently active
        if (self.manual_edit_mode and 
            self.selected_segment and 
            self.selected_segment.get('text')):
            
            current_manual_text = self.manual_text_edit.toPlainText().strip()
            last_ai_text = getattr(self, '_last_ai_text_loaded', '')
            
            # Update manual editor if it's empty or contains the previously loaded AI text
            if not current_manual_text or current_manual_text == last_ai_text:
                self.manual_text_edit.setPlainText(self.selected_segment['text'])
                self._last_ai_text_loaded = self.selected_segment['text']
                self.update_word_count()
        
        # Update play segment button
        if hasattr(self, 'play_segment_button'):
            if self.manual_edit_mode:
                self.play_segment_button.setText("▶ Play Segment (Manual Mode)")
            else:
                model_name = self.selected_segment['model']
                self.play_segment_button.setText(f"▶ Play {model_name} Segment")
    
    def get_selected_segment(self):
        """Get the selected segment (either AI or manual)"""
        if self.manual_edit_mode:
            # Create a custom segment with manual text
            manual_text = self.manual_text_edit.toPlainText().strip()
            if not manual_text:
                # If no manual text, fall back to AI selection or original
                return self.selected_segment if self.selected_segment else self.current_selection
            
            # Create manual segment based on original timing
            base_segment = self.current_selection if self.current_selection else self.segments[0]
            manual_segment = {
                'model': 'Manual Edit',
                'start': base_segment['start'],
                'end': base_segment['end'],
                'text': manual_text,
                'is_base': False,
                'is_manual': True  # Flag to identify manual edits
            }
            return manual_segment
        else:
            # Return selected AI transcription
            return self.selected_segment if self.selected_segment else self.current_selection
    
    def accept(self):
        """Override accept to validate manual input"""
        if self.manual_edit_mode:
            manual_text = self.manual_text_edit.toPlainText().strip()
            if not manual_text:
                reply = QMessageBox.question(self, "Empty Manual Transcription", 
                                           "You haven't entered any text in the manual editor. "
                                           "Do you want to proceed with an empty transcription?",
                                           QMessageBox.Yes | QMessageBox.No, 
                                           QMessageBox.No)
                if reply == QMessageBox.No:
                    return
        
        super().accept()
    
    def closeEvent(self, event):
        """Handle dialog closing with proper cleanup"""
        # Clean up audio player - check if it exists first
        if hasattr(self, 'audio_player') and self.audio_player:
            try:
                self.audio_player.cleanup()
            except Exception as e:
                print(f"Error cleaning up audio player: {e}")
        
        event.accept()
    
    def on_tab_changed(self, index):
        """Handle tab change between AI transcriptions and manual editor"""
        if index == 1:  # Manual editor tab
            self.manual_edit_mode = True
            
            # Auto-populate manual editor with currently selected AI transcription
            if self.selected_segment and self.selected_segment.get('text'):
                current_manual_text = self.manual_text_edit.toPlainText().strip()
                # Only update if manual editor is empty or if it contains old AI text
                if not current_manual_text or not hasattr(self, '_last_ai_text_loaded'):
                    self.manual_text_edit.setPlainText(self.selected_segment['text'])
                    self._last_ai_text_loaded = self.selected_segment['text']
            
            self.update_selection_status()
            self.update_word_count()
            
            # Update play button for manual mode
            if hasattr(self, 'play_segment_button'):
                self.play_segment_button.setText("▶ Play Segment (Manual Mode)")
        else:  # AI transcriptions tab
            self.manual_edit_mode = False
            self.update_selection_status()
            
            # Update play button based on selected AI model
            if hasattr(self, 'play_segment_button') and self.selected_segment:
                model_name = self.selected_segment['model']
                self.play_segment_button.setText(f"▶ Play {model_name} Segment")
    
    def clear_manual_text(self):
        """Clear the manual text editor"""
        self.manual_text_edit.clear()
        self.update_word_count()
    
    def copy_from_ai_selection(self):
        """Copy text from currently selected AI transcription to manual editor"""
        if self.selected_segment and self.selected_segment.get('text'):
            self.manual_text_edit.setPlainText(self.selected_segment['text'])
            self._last_ai_text_loaded = self.selected_segment['text']
            self.update_word_count()
        else:
            QMessageBox.information(self, "No AI Selection", 
                                  "Please select an AI transcription first, then click this button to copy its text.")
    
    def update_word_count(self):
        """Update the word count display"""
        text = self.manual_text_edit.toPlainText()
        word_count = len(text.split()) if text.strip() else 0
        char_count = len(text)
        self.word_count_label.setText(f"Words: {word_count} | Characters: {char_count}")
    
    def update_selection_status(self):
        """Update the selection status label"""
        if self.manual_edit_mode:
            text = self.manual_text_edit.toPlainText().strip()
            if text:
                preview = text[:50] + "..." if len(text) > 50 else text
                self.selection_status_label.setText(f"📝 Manual transcription: \"{preview}\"")
                self.selection_status_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #e8f5e8; border-radius: 3px; color: #2d5a2d;")
            else:
                self.selection_status_label.setText("📝 Manual editor active (no text entered)")
                self.selection_status_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #fff3cd; border-radius: 3px; color: #856404;")
        else:
            if self.selected_segment:
                model_name = self.selected_segment['model']
                preview = self.selected_segment['text'][:50] + "..." if len(self.selected_segment['text']) > 50 else self.selected_segment['text']
                self.selection_status_label.setText(f"🤖 AI selection ({model_name}): \"{preview}\"")
                self.selection_status_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #e3f2fd; border-radius: 3px; color: #1565c0;")
            else:
                self.selection_status_label.setText("No transcription selected")
                self.selection_status_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #f8f9fa; border-radius: 3px; color: #6c757d;")
    
    def on_radio_selection_changed(self, button):
        """Handle radio button selection change"""
        index = self.button_group.id(button)
        self.selected_segment = self.segments[index]
        self.update_all_frame_selections()
        self.update_selection_status()
        
        # Auto-update manual editor if it's currently active and empty or contains old AI text
        if (self.manual_edit_mode and 
            self.selected_segment and 
            self.selected_segment.get('text')):
            
            current_manual_text = self.manual_text_edit.toPlainText().strip()
            last_ai_text = getattr(self, '_last_ai_text_loaded', '')
            
            # Update manual editor if it's empty or contains the previously loaded AI text
            if not current_manual_text or current_manual_text == last_ai_text:
                self.manual_text_edit.setPlainText(self.selected_segment['text'])
                self._last_ai_text_loaded = self.selected_segment['text']
                self.update_word_count()
        
        # Update play button text
        if hasattr(self, 'play_segment_button'):
            if self.manual_edit_mode:
                self.play_segment_button.setText("▶ Play Segment (Manual Mode)")
            else:
                model_name = self.selected_segment['model']
                self.play_segment_button.setText(f"▶ Play {model_name} Segment")
    
    def update_all_frame_selections(self):
        """Update visual selection state for all frames"""
        for i, frame in enumerate(self.clickable_frames):
            radio = self.button_group.button(i)
            is_selected = radio and radio.isChecked()
            self.update_frame_selection(i, is_selected)
    
    def update_frame_selection(self, index, selected):
        """Update visual selection state for a specific frame"""
        frame = self.clickable_frames[index]
        segment = self.segments[index]
        
        if selected:
            # Selected state
            if segment['is_base']:
                frame.setStyleSheet("""
                    ClickableFrame { 
                        background-color: #d0e8ff; 
                        border: 3px solid #2196F3;
                        border-radius: 5px;
                        margin: 2px;
                    }
                    ClickableFrame:hover { 
                        background-color: #c0d8ef; 
                        border: 3px solid #1976D2;
                    }
                """)
            else:
                frame.setStyleSheet("""
                    ClickableFrame { 
                        background-color: #ffe0c0; 
                        border: 3px solid #FF9800;
                        border-radius: 5px;
                        margin: 2px;
                    }
                    ClickableFrame:hover { 
                        background-color: #ffd0a0; 
                        border: 3px solid #F57C00;
                    }
                """)
        else:
            # Unselected state
            if segment['is_base']:
                frame.setStyleSheet("""
                    ClickableFrame { 
                        background-color: #f0f8ff; 
                        border: 2px solid #ccc;
                        border-radius: 5px;
                        margin: 2px;
                    }
                    ClickableFrame:hover { 
                        background-color: #e0f0ff; 
                        border: 2px solid #2196F3;
                    }
                """)
            else:
                frame.setStyleSheet("""
                    ClickableFrame { 
                        background-color: #fff8f0; 
                        border: 2px solid #ccc;
                        border-radius: 5px;
                        margin: 2px;
                    }
                    ClickableFrame:hover { 
                        background-color: #ffe8d0; 
                        border: 2px solid #FF9800;
                    }
                """)

class AudioPlayerWidget(QWidget):
    """Professional audio player widget with VLC-like controls and proper cleanup"""
    
    # Signals for player events
    position_changed = pyqtSignal(float)  # Position in seconds
    duration_changed = pyqtSignal(float)  # Duration in seconds
    playback_started = pyqtSignal()
    playback_paused = pyqtSignal() 
    playback_stopped = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_file = None
        self.duration = 0.0
        self.position = 0.0
        self.is_playing = False
        self.is_seeking = False
        
        # Initialize VLC components as None first
        self.vlc_instance = None
        self.media_player = None
        self.event_manager = None
        self.update_timer = None
        
        # Initialize VLC with proper error handling
        if VLC_AVAILABLE:
            try:
                # Create VLC instance with minimal output to avoid GUI conflicts
                self.vlc_instance = vlc.Instance([
                    '--intf', 'dummy',          # No interface
                    '--no-video',               # Audio only
                    '--no-xlib',                # No X11 (Linux)
                    '--quiet',                  # Reduce output
                    '--no-stats',               # No statistics
                    '--no-media-library',       # No media library
                ])
                self.media_player = self.vlc_instance.media_player_new()
                self.vlc_available = True
                
                # Set up event manager for position tracking
                self.event_manager = self.media_player.event_manager()
                self.event_manager.event_attach(vlc.EventType.MediaPlayerTimeChanged, self.on_time_changed)
                self.event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, self.on_end_reached)
                self.event_manager.event_attach(vlc.EventType.MediaPlayerLengthChanged, self.on_length_changed)
                
            except Exception as e:
                print(f"Failed to initialize VLC: {e}")
                self.vlc_available = False
                self.vlc_instance = None
                self.media_player = None
                self.event_manager = None
        else:
            self.vlc_available = False
        
        # Set up UI
        self.setup_ui()
        
        # Timer for UI updates when VLC events don't fire properly
        if self.vlc_available:
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self.update_position)
            self.update_timer.setInterval(100)  # Update every 100ms
    
    def setup_ui(self):
        """Set up the audio player UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Audio file info
        self.file_label = QLabel("No audio file loaded")
        self.file_label.setStyleSheet("font-weight: bold; color: #333; padding: 5px;")
        layout.addWidget(self.file_label)
        
        # Progress bar (time slider)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(1000)  # We'll use percentage-based positioning
        self.time_slider.setValue(0)
        self.time_slider.setEnabled(False)
        self.time_slider.sliderPressed.connect(self.on_slider_pressed)
        self.time_slider.sliderReleased.connect(self.on_slider_released)
        self.time_slider.sliderMoved.connect(self.on_slider_moved)
        layout.addWidget(self.time_slider)
        
        # Time labels
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.current_time_label.setMinimumWidth(40)
        self.total_time_label = QLabel("00:00")
        self.total_time_label.setMinimumWidth(40)
        
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)
        layout.addLayout(time_layout)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        # Play/Pause button
        self.play_pause_button = QPushButton()
        self.play_pause_button.setFixedSize(40, 40)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.play_pause_button.setEnabled(False)
        
        # Stop button
        self.stop_button = QPushButton()
        self.stop_button.setFixedSize(35, 35)
        self.stop_button.clicked.connect(self.stop_playback)
        self.stop_button.setEnabled(False)
        
        # Skip backward button (-10s)
        self.skip_back_button = QPushButton()
        self.skip_back_button.setFixedSize(35, 35)
        self.skip_back_button.clicked.connect(lambda: self.skip_time(-10))
        self.skip_back_button.setEnabled(False)
        self.skip_back_button.setToolTip("Skip back 10 seconds")
        
        # Skip forward button (+10s)
        self.skip_forward_button = QPushButton()
        self.skip_forward_button.setFixedSize(35, 35)
        self.skip_forward_button.clicked.connect(lambda: self.skip_time(10))
        self.skip_forward_button.setEnabled(False)
        self.skip_forward_button.setToolTip("Skip forward 10 seconds")
        
        # Volume slider
        volume_layout = QVBoxLayout()
        volume_layout.setSpacing(2)
        volume_label = QLabel("Vol")
        volume_label.setAlignment(Qt.AlignCenter)
        volume_label.setStyleSheet("font-size: 10px; color: #666;")
        
        self.volume_slider = QSlider(Qt.Vertical)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(70)
        self.volume_slider.setFixedHeight(60)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.volume_slider.setEnabled(False)
        
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_slider)
        
        # Add all controls to layout
        controls_layout.addStretch()
        controls_layout.addWidget(self.skip_back_button)
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.skip_forward_button)
        controls_layout.addStretch()
        controls_layout.addLayout(volume_layout)
        
        layout.addLayout(controls_layout)
        
        # Set button icons/text
        self.update_button_icons()
        
        # Status label
        self.status_label = QLabel("Ready" if self.vlc_available else "VLC not available - audio playback disabled")
        self.status_label.setStyleSheet("color: #666; font-size: 10px; padding: 2px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
    
    def load_audio_file(self, file_path):
        """Load an audio file for playback"""
        if not self.vlc_available:
            self.status_label.setText("VLC not available - cannot load audio")
            return False
        
        if not os.path.exists(file_path):
            self.status_label.setText(f"Audio file not found: {os.path.basename(file_path)}")
            return False
        
        try:
            # Stop any current playback
            self.stop_playback()
            
            # Load new media
            self.audio_file = file_path
            media = self.vlc_instance.media_new(file_path)
            self.media_player.set_media(media)
            
            # Update UI
            self.file_label.setText(f"♪ {os.path.basename(file_path)}")
            self.status_label.setText("Audio loaded - ready to play")
            
            # Enable controls
            self.play_pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.skip_back_button.setEnabled(True)
            self.skip_forward_button.setEnabled(True)
            self.time_slider.setEnabled(True)
            self.volume_slider.setEnabled(True)
            
            # Set initial volume
            self.set_volume(self.volume_slider.value())
            
            return True
            
        except Exception as e:
            self.status_label.setText(f"Error loading audio: {str(e)}")
            return False
    
    def toggle_playback(self):
        """Toggle between play and pause"""
        if not self.vlc_available or not self.audio_file:
            return
        
        try:
            # Check actual VLC state to sync with our internal state
            vlc_state = self.media_player.get_state()
            vlc_is_playing = vlc_state in [vlc.State.Playing, vlc.State.Buffering]
            
            # Sync our state with VLC's actual state
            if vlc_is_playing and not self.is_playing:
                self.is_playing = True
                self.update_button_icons()
            elif not vlc_is_playing and self.is_playing:
                self.is_playing = False
                self.update_button_icons()
            
            # Now toggle based on synchronized state
            if self.is_playing:
                self.pause_playback()
            else:
                self.start_playback()
        except Exception as e:
            self.status_label.setText(f"Playback error: {str(e)}")

    def start_playback(self):
        """Start audio playback"""
        if not self.vlc_available or not self.audio_file:
            return
        
        try:
            # Check if we're at or near the end, and reset to beginning if so
            current_time = self.media_player.get_time()
            length = self.media_player.get_length()
            
            if length > 0 and current_time >= (length - 1000):  # Within 1 second of end
                self.media_player.set_time(0)  # Reset to beginning
                self.position = 0.0
                self.time_slider.setValue(0)
                self.current_time_label.setText("00:00")
            
            self.media_player.play()
            self.is_playing = True
            self.update_button_icons()
            if self.update_timer:
                self.update_timer.start()
            self.status_label.setText("Playing...")
            self.playback_started.emit()
        except Exception as e:
            self.status_label.setText(f"Play error: {str(e)}")
    
    def pause_playback(self):
        """Pause audio playback"""
        if not self.vlc_available:
            return
        
        try:
            self.media_player.pause()
            self.is_playing = False
            self.update_button_icons()
            if self.update_timer:
                self.update_timer.stop()
            self.status_label.setText("Paused")
            self.playback_paused.emit()
        except Exception as e:
            self.status_label.setText(f"Pause error: {str(e)}")
    
    def stop_playback(self):
        """Stop audio playback"""
        if not self.vlc_available:
            return
        
        try:
            self.media_player.stop()
            self.is_playing = False
            self.position = 0.0
            self.update_button_icons()
            if self.update_timer:
                self.update_timer.stop()
            self.time_slider.setValue(0)
            self.current_time_label.setText("00:00")
            self.status_label.setText("Stopped")
            self.playback_stopped.emit()
        except Exception as e:
            self.status_label.setText(f"Stop error: {str(e)}")

    def skip_time(self, seconds):
        """Skip forward or backward by specified seconds"""
        if not self.vlc_available or not self.audio_file:
            return
        
        try:
            current_time = self.media_player.get_time()  # Time in milliseconds
            new_time = max(0, current_time + (seconds * 1000))
            
            # Don't skip beyond the end
            if self.duration > 0:
                max_time = self.duration * 1000
                new_time = min(new_time, max_time)
            
            self.media_player.set_time(int(new_time))
            self.status_label.setText(f"Skipped {seconds:+d}s")
        except Exception as e:
            self.status_label.setText(f"Skip error: {str(e)}")
    
    def set_volume(self, volume):
        """Set playback volume (0-100)"""
        if not self.vlc_available:
            return
        
        try:
            self.media_player.audio_set_volume(volume)
        except Exception as e:
            self.status_label.setText(f"Volume error: {str(e)}")
    
    def on_slider_pressed(self):
        """Handle slider press (start seeking)"""
        self.is_seeking = True
    
    def on_slider_released(self):
        """Handle slider release (end seeking and set position)"""
        if not self.vlc_available or not self.audio_file or self.duration <= 0:
            self.is_seeking = False
            return
        
        try:
            # Calculate new position
            slider_value = self.time_slider.value()
            new_position = (slider_value / 1000.0) * self.duration

            # Set media player position (with padding offset)
            self.media_player.set_time(int(new_position * 1000))
            
            self.is_seeking = False
        except Exception as e:
            self.status_label.setText(f"Seek error: {str(e)}")
            self.is_seeking = False

    def on_slider_moved(self, value):
        """Handle slider movement (preview time)"""
        if self.duration > 0:
            # FIXED: Calculate preview time based on content duration (without padding)
            content_duration = max(0, self.duration - 2.0)  # Remove padding from both ends
            if content_duration > 0:
                preview_time = (value / 1000.0) * content_duration
                self.current_time_label.setText(self.format_time(preview_time))
    
    def update_position(self):
        """Update position display (called by timer)"""
        if not self.vlc_available or not self.audio_file or self.is_seeking:
            return
        
        try:
            # Get current time and duration
            current_time = self.media_player.get_time()  # milliseconds
            length = self.media_player.get_length()      # milliseconds
            
            if current_time >= 0 and length > 0:
                # FIXED: Subtract 1 second padding from display time
                raw_position = current_time / 1000.0
                display_position = max(0, raw_position - 1.0)  # Show content time, not file time
                
                raw_duration = length / 1000.0
                display_duration = max(0, raw_duration - 2.0)  # Subtract padding from both ends
                
                # Store raw values for calculations
                self.position = raw_position
                self.duration = raw_duration
                
                # Update slider based on content time
                if display_duration > 0:
                    slider_position = int((display_position / display_duration) * 1000)
                    self.time_slider.setValue(slider_position)
                
                # Update time labels to show content time
                self.current_time_label.setText(self.format_time(display_position))
                self.total_time_label.setText(self.format_time(display_duration))
                
                # Emit signals with display time (content time, not raw file time)
                self.position_changed.emit(display_position)
                if abs(display_duration - (length/1000.0 - 2.0)) > 0.1:  # Duration changed significantly
                    self.duration_changed.emit(display_duration)
                    
        except Exception as e:
            pass  # Ignore errors during position updates

    def update_button_icons(self):
        """Update button icons and text"""
        # Try to use system icons, fall back to text
        style = self.style()
        
        # Play/Pause button
        if self.is_playing:
            icon = style.standardIcon(QStyle.SP_MediaPause)
            if icon.isNull():
                self.play_pause_button.setText("⏸")
            else:
                self.play_pause_button.setIcon(icon)
                self.play_pause_button.setText("")
            self.play_pause_button.setToolTip("Pause")
        else:
            icon = style.standardIcon(QStyle.SP_MediaPlay)
            if icon.isNull():
                self.play_pause_button.setText("▶")
            else:
                self.play_pause_button.setIcon(icon)
                self.play_pause_button.setText("")
            self.play_pause_button.setToolTip("Play")
        
        # Stop button
        icon = style.standardIcon(QStyle.SP_MediaStop)
        if icon.isNull():
            self.stop_button.setText("⏹")
        else:
            self.stop_button.setIcon(icon)
            self.stop_button.setText("")
        self.stop_button.setToolTip("Stop")
        
        # Skip buttons
        self.skip_back_button.setText("⏪")
        self.skip_forward_button.setText("⏩")
    
    def on_time_changed(self, event):
        """VLC event: time changed"""
        # This is called from VLC's thread, so we need to be careful
        pass  # We'll rely on the timer for updates instead
    
    def on_length_changed(self, event):
        """VLC event: length/duration changed"""
        pass  # We'll rely on the timer for updates instead
    
    def on_end_reached(self, event):
        """VLC event: playback ended"""
        # This needs to be called from the main thread
        QTimer.singleShot(0, self.handle_end_reached)
    
    @pyqtSlot()
    def handle_end_reached(self):
        """Handle end of playback (called from main thread)"""
        try:
            # Actually stop the VLC player to reset its internal state
            self.media_player.stop()
        except Exception:
            pass
        
        self.is_playing = False
        self.position = 0.0
        
        self.update_button_icons()
        if self.update_timer:
            self.update_timer.stop()
        self.time_slider.setValue(0)
        self.current_time_label.setText("00:00")
        self.status_label.setText("Finished")
        self.playback_stopped.emit()

    def format_time(self, seconds):
        """Format time as MM:SS"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def get_position(self):
        """Get current playback position in seconds"""
        return self.position
    
    def get_duration(self):
        """Get total duration in seconds"""
        return self.duration
    
    def is_playback_active(self):
        """Check if audio is currently playing"""
        return self.is_playing
    
    def cleanup(self):
        """Clean up resources with proper error handling"""
        try:
            # Stop playback first
            self.stop_playback()
            
            # Stop and clean up timer
            if hasattr(self, 'update_timer') and self.update_timer:
                self.update_timer.stop()
                self.update_timer = None
            
            # Clean up VLC components
            if hasattr(self, 'media_player') and self.media_player:
                try:
                    self.media_player.release()
                except:
                    pass
                self.media_player = None
                
            if hasattr(self, 'vlc_instance') and self.vlc_instance:
                try:
                    self.vlc_instance.release()
                except:
                    pass
                self.vlc_instance = None
                
            self.event_manager = None
            
        except Exception as e:
            print(f"Error during audio player cleanup: {e}")

class ClickableFrame(QFrame):
    """Custom frame that emits clicked signal when clicked anywhere"""
    
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def mousePressEvent(self, event):
        """Handle mouse press event to emit clicked signal"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

class ModelComparisonProcessor(QThread):
    """Optimized thread for processing model comparisons - process all clips with one model before switching"""
    progress_updated = pyqtSignal(int)
    comparison_progress = pyqtSignal(str)
    comparison_finished = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, processed_clips, transcription_settings):
        super().__init__()
        self.processed_clips = processed_clips
        self.transcription_settings = transcription_settings
        self.comparison_results = []
        
    def run(self):
        """Enhanced model comparison using batch processing"""
        try:
            if not self.transcription_settings.comparison_models:
                self.error_occurred.emit("No comparison models selected")
                self.comparison_finished.emit([])
                return
            
            total_clips = len(self.processed_clips)
            comparison_models = self.transcription_settings.comparison_models
            total_operations = total_clips * (len(comparison_models) + 1)
            completed_operations = 0
            
            self.comparison_progress.emit(f"🚀 Starting enhanced comparison with {len(comparison_models)} models...")
            
            # Create enhanced transcriber
            comparison_transcriber = WhisperTranscriber(self.transcription_settings)
            
            # Store results by model
            model_results = {}
            
            # Store base model results first
            base_model = self.transcription_settings.base_model
            model_results[base_model] = {}
            
            for clip_idx, clip_info in enumerate(self.processed_clips):
                base_result = {
                    "model": base_model,
                    "segments": clip_info.get("segments", []),
                    "info": clip_info.get("transcription_info", {})
                }
                model_results[base_model][clip_idx] = base_result
                completed_operations += 1
                
                progress = int((completed_operations / total_operations) * 100)
                self.progress_updated.emit(progress)
            
            # Process each comparison model using batch processing
            for model_name in comparison_models:
                self.comparison_progress.emit(f"🔄 Processing all clips with {model_name}...")
                
                # Switch to new model
                model = comparison_transcriber.switch_model_efficiently(model_name)
                if not model:
                    self.error_occurred.emit(f"Failed to load model {model_name}")
                    continue
                
                # Collect audio paths for batch processing
                audio_paths = []
                clip_indices = []
                for clip_idx, clip_info in enumerate(self.processed_clips):
                    audio_path = clip_info.get("output_path")
                    if audio_path and os.path.exists(audio_path):
                        audio_paths.append(audio_path)
                        clip_indices.append(clip_idx)
                
                # Enhanced batch transcription
                self.comparison_progress.emit(f"🚀 Batch processing {len(audio_paths)} clips with {model_name}")
                batch_results = comparison_transcriber.batch_transcribe_clips(
                    audio_paths, 
                    model_size=model_name, 
                    timeout=self.transcription_settings.timeout_seconds
                )
                
                # Store results
                model_results[model_name] = {}
                for i, (clip_idx, (segments, info, success)) in enumerate(zip(clip_indices, batch_results)):
                    clip_info = self.processed_clips[clip_idx]
                    
                    if success:
                        # Adjust segments for padding
                        for segment in segments:
                            segment.start -= 1.0
                            segment.end -= 1.0
                            segment.start = max(0, segment.start)
                            segment.end = min(clip_info["duration"], segment.end)
                        
                        model_results[model_name][clip_idx] = {
                            "model": model_name,
                            "segments": segments,
                            "info": info
                        }
                    else:
                        model_results[model_name][clip_idx] = {
                            "model": model_name,
                            "segments": [],
                            "info": {}
                        }
                    
                    completed_operations += 1
                    progress = int((completed_operations / total_operations) * 100)
                    self.progress_updated.emit(progress)
                
                # Clear model cache
                comparison_transcriber.clear_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Create comparison results
            self.comparison_progress.emit("📊 Creating comparison results...")
            for clip_idx, clip_info in enumerate(self.processed_clips):
                base_result = model_results[base_model][clip_idx]
                comparison_results = []
                
                for model_name in comparison_models:
                    if model_name in model_results and clip_idx in model_results[model_name]:
                        comparison_results.append(model_results[model_name][clip_idx])
                
                if comparison_results:
                    comp_result = ModelComparisonResult(clip_info, base_result, comparison_results)
                    self.comparison_results.append(comp_result)
            
            self.comparison_progress.emit("✅ Enhanced model comparison completed!")
            self.comparison_finished.emit(self.comparison_results)
            
        except Exception as e:
            self.error_occurred.emit(f"Error during enhanced model comparison: {str(e)}")
            self.comparison_finished.emit([])

class VocalIsolationSettings:
    """Store vocal isolation settings"""
    def __init__(self):
        self.enabled = False
        self.model = "htdemucs"
        self.extract_vocals = True
        self.extract_instruments = False
        self.pre_download = True
        self.use_custom_cache = True

class TranscriptionSettings:
    """Store transcription settings"""
    def __init__(self):
        self.enabled = True
        self.model_size = "large-v2"
        self.language = "en"
        self.device = "cuda"
        self.beam_size = 5
        self.create_srt = True
        self.create_combined_srt = True
        # Add these missing attributes:
        self.fast_mode = False
        self.enable_fallbacks = True
        self.max_fallback_attempts = 3
        self.timeout_seconds = 30
        self.fallback_models = ["large-v3"]
        # Add new attributes for improved alignment
        self.use_improved_alignment = True
        self.align_model = "WAV2VEC2_ASR_LARGE_LV60K_960H"
        self.batch_size = 8
        self.compute_type = "int8"  # Changed from hard-coded to a setting
        
        # Add model comparison settings
        self.enable_model_comparison = False
        self.comparison_models = []  # List of models to compare against base model
        self.base_model = "large-v2"  # Base model for comparison
        
        # NEW: Add voting system settings
        self.enable_voting_system = False
        self.voting_models = [
            "tiny", "tiny.en", 
            "base", "base.en",
            "small", "small.en", 
            "medium", "medium.en",
            "large-v1", "large-v2", "large-v3", "large-v3-turbo",
            "turbo", "distil-medium.en", "distil-small.en",
            "distil-large-v2", "distil-large-v3"
        ]  # All available models for voting
        self.voting_confidence_threshold = 0.6  # Minimum confidence for word voting
        self.voting_agreement_threshold = 0.5   # Minimum agreement ratio for accepting a word
        self.voting_prefer_english = True       # Prefer English transcriptions when available
        self.voting_english_bias = 1.2          # Boost score for English-specific models

class VotingResult:
    """Class to store voting results for word-level comparison"""
    def __init__(self, clip_info, model_results):
        self.clip_info = clip_info
        self.model_results = model_results  # List of {model, segments, word_data}
        self.voted_segments = []
        self.voting_statistics = {}
        
        # Perform word-level voting
        self._perform_word_voting()
        
    def _perform_word_voting(self):
        """Perform word-by-word voting across all models with English preference"""
        print(f"Performing word-level voting for {len(self.model_results)} model results")
        
        # Set English preference flag for voting methods
        self._english_preference = True
        
        # Step 1: Align all segments by time to create voting blocks
        time_aligned_blocks = self._create_time_aligned_blocks()
        
        # Step 2: For each time block, vote on the best words
        self.voted_segments = []
        total_words = 0
        voted_words = 0
        model_agreement_counts = {result['model']: 0 for result in self.model_results}
        english_model_wins = 0
        multilingual_detections = 0
        
        for block in time_aligned_blocks:
            voted_segment = self._vote_on_block(block)
            if voted_segment:
                self.voted_segments.append(voted_segment)
                # Count statistics with language tracking
                if hasattr(voted_segment, 'voting_info'):
                    for word_info in voted_segment.voting_info:
                        total_words += 1
                        if word_info.get('voted'):
                            voted_words += 1
                            winning_model = word_info.get('winning_model')
                            if winning_model in model_agreement_counts:
                                model_agreement_counts[winning_model] += 1
                            
                            # Track English model preferences
                            if winning_model.endswith('.en'):
                                english_model_wins += 1
                            
                            # Track potential multilingual content
                            if word_info.get('language_score', 0) < 0.05:  # Low English confidence
                                multilingual_detections += 1
        
        # Store voting statistics with language info
        self.voting_statistics = {
            'total_words': total_words,
            'voted_words': voted_words,
            'model_agreement': model_agreement_counts,
            'voting_success_rate': voted_words / total_words if total_words > 0 else 0,
            'english_model_wins': english_model_wins,
            'english_preference_rate': english_model_wins / voted_words if voted_words > 0 else 0,
            'multilingual_detections': multilingual_detections,
            'multilingual_rate': multilingual_detections / total_words if total_words > 0 else 0
        }
        
        print(f"Voting completed: {voted_words}/{total_words} words voted successfully")
        print(f"English models won: {english_model_wins}/{voted_words} words ({english_model_wins/voted_words*100:.1f}%)")
        if multilingual_detections > 0:
            print(f"Potential multilingual content detected: {multilingual_detections} words ({multilingual_detections/total_words*100:.1f}%)")
            
        for model, count in model_agreement_counts.items():
            if count > 0:
                percentage = count/voted_words*100 if voted_words > 0 else 0
                english_indicator = " (EN)" if model.endswith('.en') else ""
                print(f"  {model}{english_indicator}: {count} words ({percentage:.1f}%)")
    
    def _create_time_aligned_blocks(self):
        """Create time-aligned blocks where models agree on timing"""
        all_segments = []
        
        # Collect all segments with model info
        for model_result in self.model_results:
            model_name = model_result['model']
            for segment in model_result['segments']:
                all_segments.append({
                    'model': model_name,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'words': getattr(segment, 'words', []),
                    'segment_obj': segment
                })
        
        # Sort by start time
        all_segments.sort(key=lambda x: x['start'])
        
        # Group segments that overlap in time
        time_blocks = []
        current_block = []
        
        for segment in all_segments:
            if not current_block:
                current_block = [segment]
            else:
                # Check if this segment overlaps with any in current block
                overlaps = False
                for block_segment in current_block:
                    if (segment['start'] < block_segment['end'] and 
                        segment['end'] > block_segment['start']):
                        overlaps = True
                        break
                
                if overlaps or segment['start'] - current_block[-1]['end'] < 1.0:
                    current_block.append(segment)
                else:
                    # Start new block
                    if current_block:
                        time_blocks.append(current_block)
                    current_block = [segment]
        
        # Add final block
        if current_block:
            time_blocks.append(current_block)
        
        return time_blocks
    
    def _vote_on_block(self, block_segments):
        """Vote on the best transcription for a time block"""
        if not block_segments:
            return None
        
        # Find the time span covered by this block
        block_start = min(seg['start'] for seg in block_segments)
        block_end = max(seg['end'] for seg in block_segments)
        
        # Collect all words from all models in this time block
        all_words_by_model = {}
        for segment in block_segments:
            model = segment['model']
            if model not in all_words_by_model:
                all_words_by_model[model] = []
            
            # Extract words (with timing if available)
            if segment['words']:
                # Word-level timing available
                for word in segment['words']:
                    all_words_by_model[model].append({
                        'word': word.get('word', '').strip(),
                        'start': word.get('start', segment['start']),
                        'end': word.get('end', segment['end']),
                        'confidence': word.get('probability', 0.0)
                    })
            else:
                # No word-level timing, split text
                words = segment['text'].split()
                word_duration = (segment['end'] - segment['start']) / len(words) if words else 0
                for i, word in enumerate(words):
                    word_start = segment['start'] + i * word_duration
                    word_end = word_start + word_duration
                    all_words_by_model[model].append({
                        'word': word.strip(),
                        'start': word_start,
                        'end': word_end,
                        'confidence': 0.5  # Default confidence
                    })
        
        # Perform word-by-word voting
        voted_words = self._vote_on_words(all_words_by_model, block_start, block_end)
        
        if not voted_words:
            return None
        
        # Create final segment from voted words
        final_text = ' '.join(word['word'] for word in voted_words)
        
        # Create a segment-like object
        class VotedSegment:
            def __init__(self, start, end, text, voting_info):
                self.start = start
                self.end = end
                self.text = text
                self.voting_info = voting_info
        
        return VotedSegment(block_start, block_end, final_text, voted_words)
    
    def _vote_on_words(self, words_by_model, block_start, block_end):
        """Vote on individual words across models"""
        # Create time slots for word alignment
        time_slots = []
        slot_duration = 0.1  # 100ms slots
        
        # Collect all words with their time slots
        word_candidates = []
        for model, words in words_by_model.items():
            for word in words:
                start_slot = int((word['start'] - block_start) / slot_duration)
                end_slot = int((word['end'] - block_start) / slot_duration)
                word_candidates.append({
                    'model': model,
                    'word': word['word'],
                    'start_slot': start_slot,
                    'end_slot': end_slot,
                    'confidence': word['confidence'],
                    'original_start': word['start'],
                    'original_end': word['end']
                })
        
        # Group candidates by time slot overlap
        word_groups = self._group_words_by_overlap(word_candidates)
        
        # Vote on each group
        voted_words = []
        for group in word_groups:
            voted_word = self._vote_on_word_group(group)
            if voted_word:
                voted_words.append(voted_word)
        
        return voted_words
    
    def _group_words_by_overlap(self, word_candidates):
        """Group words that overlap in time"""
        groups = []
        remaining_words = word_candidates.copy()
        
        while remaining_words:
            current_word = remaining_words.pop(0)
            current_group = [current_word]
            
            # Find all words that overlap with current word
            i = 0
            while i < len(remaining_words):
                word = remaining_words[i]
                
                # Check for overlap
                if (word['start_slot'] <= current_word['end_slot'] and 
                    word['end_slot'] >= current_word['start_slot']):
                    current_group.append(word)
                    remaining_words.pop(i)
                else:
                    i += 1
            
            groups.append(current_group)
        
        return groups
    
    def _vote_on_word_group(self, word_group):
        """Vote on the best word from a group of candidates with English preference"""
        if not word_group:
            return None
        
        # Count votes for each unique word (case-insensitive)
        word_votes = {}
        for candidate in word_group:
            word_lower = candidate['word'].lower()
            if word_lower not in word_votes:
                word_votes[word_lower] = []
            word_votes[word_lower].append(candidate)
        
        # Find the word with the most votes and highest confidence
        best_word = None
        best_score = 0
        
        total_models = len(set(candidate['model'] for candidate in word_group))
        agreement_threshold = max(1, int(total_models * 0.5))  # At least 50% agreement
        
        for word_text, candidates in word_votes.items():
            if len(candidates) >= agreement_threshold:
                # Calculate average confidence
                avg_confidence = sum(c['confidence'] for c in candidates) / len(candidates)
                
                # Apply English model bias if enabled
                english_bonus = 0
                if hasattr(self, '_english_preference') and self._english_preference:
                    english_models = sum(1 for c in candidates if c['model'].endswith('.en'))
                    if english_models > 0:
                        english_ratio = english_models / len(candidates)
                        english_bonus = english_ratio * 0.15  # 15% bonus for English models
                
                # Enhanced scoring with English preference and language detection
                base_score = len(candidates) * 0.6 + avg_confidence * 0.3
                language_score = self._calculate_language_score(word_text, candidates)
                
                # Final score combines vote count, confidence, English bias, and language detection
                score = base_score + english_bonus + language_score
                
                if score > best_score:
                    best_score = score
                    best_word = {
                        'word': candidates[0]['word'],  # Use original case from first candidate
                        'vote_count': len(candidates),
                        'confidence': avg_confidence,
                        'models': [c['model'] for c in candidates],
                        'start': min(c['original_start'] for c in candidates),
                        'end': max(c['original_end'] for c in candidates),
                        'voted': True,
                        'winning_model': max(candidates, key=lambda x: x['confidence'])['model'],
                        'english_bonus': english_bonus,
                        'language_score': language_score,
                        'final_score': score
                    }
        
        # If no clear winner, use the highest confidence word with English preference
        if not best_word:
            # Sort candidates by English preference first, then confidence
            def sort_key(candidate):
                english_boost = 0.1 if candidate['model'].endswith('.en') else 0
                language_boost = self._calculate_single_word_language_score(candidate['word'])
                return candidate['confidence'] + english_boost + language_boost
            
            best_candidate = max(word_group, key=sort_key)
            best_word = {
                'word': best_candidate['word'],
                'vote_count': 1,
                'confidence': best_candidate['confidence'],
                'models': [best_candidate['model']],
                'start': best_candidate['original_start'],
                'end': best_candidate['original_end'],
                'voted': False,
                'winning_model': best_candidate['model'],
                'english_bonus': 0.1 if best_candidate['model'].endswith('.en') else 0,
                'language_score': self._calculate_single_word_language_score(best_candidate['word']),
                'final_score': best_candidate['confidence']
            }
        
        return best_word
    
    def _calculate_language_score(self, word_text, candidates):
        """Calculate language preference score for a word"""
        # Simple heuristics for English vs other languages
        english_indicators = 0
        total_candidates = len(candidates)
        
        # Check if word looks English
        if self._looks_like_english(word_text):
            english_indicators += 0.1
        
        # Prefer results from English-specific models
        english_model_count = sum(1 for c in candidates if c['model'].endswith('.en'))
        if english_model_count > 0:
            english_model_ratio = english_model_count / total_candidates
            english_indicators += english_model_ratio * 0.1
        
        # Check for common English patterns
        if self._has_english_patterns(word_text):
            english_indicators += 0.05
        
        return english_indicators
    
    def _calculate_single_word_language_score(self, word_text):
        """Calculate language score for a single word"""
        score = 0
        
        if self._looks_like_english(word_text):
            score += 0.1
        
        if self._has_english_patterns(word_text):
            score += 0.05
        
        return score
    
    def _looks_like_english(self, word):
        """Simple heuristic to check if a word looks English"""
        if not word or len(word) < 2:
            return True  # Short words, assume English
        
        word_lower = word.lower()
        
        # Common English words (high confidence these are English)
        common_english = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how',
            'what', 'who', 'which', 'whose', 'whom', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'a', 'an', 'some', 'any', 'all', 'each', 'every', 'many', 'much', 'few', 'little',
            'yes', 'no', 'not', 'very', 'too', 'so', 'just', 'only', 'also', 'even', 'still'
        }
        
        if word_lower in common_english:
            return True
        
        # Check for non-English characters (accented characters, non-Latin scripts)
        import re
        if re.search(r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', word_lower):
            return False  # Contains accented characters, likely not English
        
        if re.search(r'[^\x00-\x7F]', word):
            return False  # Contains non-ASCII characters, likely not English
        
        return True  # Default to English if unsure
    
    def _has_english_patterns(self, word):
        """Check for common English spelling patterns"""
        if not word or len(word) < 3:
            return True
        
        word_lower = word.lower()
        
        # Common English endings
        english_endings = [
            'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 'ful', 'less',
            'able', 'ible', 'ous', 'ious', 'eous', 'ive', 'ative', 'itive'
        ]
        
        for ending in english_endings:
            if word_lower.endswith(ending):
                return True
        
        # Common English prefixes
        english_prefixes = [
            'un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'out', 'up', 'in', 'im'
        ]
        
        for prefix in english_prefixes:
            if word_lower.startswith(prefix) and len(word_lower) > len(prefix):
                return True
        
        return False

class ModelVotingProcessor(QThread):
    """Thread for processing model voting - runs all available models"""
    progress_updated = pyqtSignal(int)
    voting_progress = pyqtSignal(str)
    voting_finished = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, processed_clips, transcription_settings):
        super().__init__()
        self.processed_clips = processed_clips
        self.transcription_settings = transcription_settings
        self.voting_results = []
        
    def run(self):
        """Process model voting with all available models"""
        try:
            if not self.transcription_settings.voting_models:
                self.error_occurred.emit("No voting models configured")
                self.voting_finished.emit([])
                return
            
            total_clips = len(self.processed_clips)
            voting_models = self.transcription_settings.voting_models
            total_operations = total_clips * len(voting_models)
            completed_operations = 0
            
            self.voting_progress.emit(f"Starting voting system with {len(voting_models)} models...")
            
            # Create transcriber for voting models
            voting_transcriber = WhisperTranscriber(self.transcription_settings)
            
            # Store voting results structure
            model_results = {}
            
            # Initialize results for each model
            for model_name in voting_models:
                model_results[model_name] = {}
            
            # Process each voting model for ALL clips before switching models
            for model_name in voting_models:
                self.voting_progress.emit(f"Loading model {model_name}...")
                
                # Efficiently switch to the new model
                model = voting_transcriber.switch_model_efficiently(model_name)
                
                if not model:
                    self.error_occurred.emit(f"Failed to load model {model_name}")
                    continue
                
                self.voting_progress.emit(f"Processing all clips with {model_name}...")
                
                # Collect all audio paths for batch processing
                audio_paths = []
                clip_indices = []
                for clip_idx, clip_info in enumerate(self.processed_clips):
                    audio_path = clip_info.get("output_path")
                    if audio_path and os.path.exists(audio_path):
                        audio_paths.append(audio_path)
                        clip_indices.append(clip_idx)
                
                # Batch transcribe all clips with this model
                batch_results = voting_transcriber.batch_transcribe_clips(
                    audio_paths, 
                    model_size=model_name, 
                    timeout=self.transcription_settings.timeout_seconds
                )
                
                # Process results
                for i, (clip_idx, (segments, info, success)) in enumerate(zip(clip_indices, batch_results)):
                    clip_info = self.processed_clips[clip_idx]
                    
                    if success:
                        # Adjust segments for padding (same as base processing)
                        for segment in segments:
                            segment.start -= 1.0  # Remove padding
                            segment.end -= 1.0
                            segment.start = max(0, segment.start)
                            segment.end = min(clip_info["duration"], segment.end)
                        
                        model_results[model_name][clip_idx] = {
                            "model": model_name,
                            "segments": segments,
                            "info": info
                        }
                    else:
                        self.error_occurred.emit(f"Failed to transcribe clip {clip_idx} with {model_name}")
                        # Store empty result to maintain structure
                        model_results[model_name][clip_idx] = {
                            "model": model_name,
                            "segments": [],
                            "info": {}
                        }
                    
                    completed_operations += 1
                    
                    # Update progress
                    progress = int((completed_operations / total_operations) * 100)
                    self.progress_updated.emit(progress)
                
                self.voting_progress.emit(f"Completed all clips with {model_name}")
                
                # Clear model cache to free memory before loading next model
                voting_transcriber.clear_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Now create voting results from the organized data
            self.voting_progress.emit("Performing word-level voting...")
            
            for clip_idx, clip_info in enumerate(self.processed_clips):
                # Collect all model results for this clip
                clip_model_results = []
                for model_name in voting_models:
                    if model_name in model_results and clip_idx in model_results[model_name]:
                        model_result = model_results[model_name][clip_idx]
                        if model_result["segments"]:  # Only include models that produced results
                            clip_model_results.append(model_result)
                
                # Create voting result if we have multiple model results
                if len(clip_model_results) >= 2:  # Need at least 2 models to vote
                    voting_result = VotingResult(clip_info, clip_model_results)
                    self.voting_results.append(voting_result)
                    
                    # Log voting statistics
                    stats = voting_result.voting_statistics
                    self.voting_progress.emit(f"Clip {clip_idx}: {stats['voted_words']}/{stats['total_words']} words voted "
                                            f"({stats['voting_success_rate']*100:.1f}% success rate)")
            
            self.voting_progress.emit("Word-level voting completed!")
            self.voting_finished.emit(self.voting_results)
            
        except Exception as e:
            self.error_occurred.emit(f"Error during model voting: {str(e)}")
            import traceback
            self.error_occurred.emit(traceback.format_exc())
            self.voting_finished.emit([])

class ModelVotingDialog(QDialog):
    """Dialog for displaying voting results and allowing user review"""
    
    def __init__(self, voting_results, parent=None):
        super().__init__(parent)
        self.voting_results = voting_results
        self.setup_ui()
        self.populate_voting_results()
        
    def setup_ui(self):
        """Set up the voting results dialog UI"""
        self.setWindowTitle("Model Voting Results - Word-Level Voting")
        self.setMinimumSize(1400, 800)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel(f"Word-Level Voting Results ({len(self.voting_results)} clips)")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        # Instructions
        instructions = QLabel(
            "• Each word was voted on by multiple Whisper models\n"
            "• Green highlighting shows words with strong model agreement\n"
            "• Red highlighting shows words with weak agreement or single model votes\n"
            "• Voting statistics show which models contributed most to the final result"
        )
        instructions.setStyleSheet("color: #666; padding: 10px; background-color: #f5f5f5; border-radius: 5px;")
        layout.addWidget(instructions)
        
        # Create main table
        self.setup_main_table()
        layout.addWidget(self.main_table)
        
        # Statistics panel
        self.setup_statistics_panel()
        layout.addWidget(self.stats_panel)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.export_button = QPushButton("Export Voting Results...")
        self.export_button.clicked.connect(self.export_results)
        
        self.create_srt_button = QPushButton("Create Voted SRT Files")
        self.create_srt_button.clicked.connect(self.create_voted_srt_files)
        self.create_srt_button.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white; padding: 8px;")
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()
        button_layout.addWidget(self.create_srt_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def setup_main_table(self):
        """Set up the main voting results table"""
        self.main_table = QTreeWidget()
        self.main_table.setHeaderLabels([
            "Clip", "Voting Success Rate", "Total Words", "Voted Words", "Final Transcription Preview"
        ])
        
        # Set column widths
        self.main_table.setColumnWidth(0, 200)  # Clip name
        self.main_table.setColumnWidth(1, 120)  # Success rate
        self.main_table.setColumnWidth(2, 100)  # Total words
        self.main_table.setColumnWidth(3, 100)  # Voted words
        self.main_table.setColumnWidth(4, 600)  # Preview
        
        # Enable features
        self.main_table.setAlternatingRowColors(True)
        self.main_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.main_table.setSortingEnabled(True)
    
    def setup_statistics_panel(self):
        """Set up the statistics panel"""
        self.stats_panel = QGroupBox("Overall Voting Statistics")
        stats_layout = QVBoxLayout(self.stats_panel)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)

    def populate_voting_results(self):
        """Populate the enhanced results table with model columns, filtering identical segments"""
        self.main_content.clear()
        
        if not self.comparison_results:
            return
        
        # Collect all unique models from all results
        all_models = set()
        for result in self.comparison_results:
            all_models.add(result.base_model)
            for comp_result in result.comparison_results:
                all_models.add(comp_result["model"])
        
        # Sort models for consistent ordering
        sorted_models = sorted(list(all_models))
        self.model_columns = sorted_models
        
        # Set up headers with model columns
        headers = ["Index", "Clip", "Time"] + sorted_models + ["Actions"]
        self.main_content.setHeaderLabels(headers)
        
        # Set column widths
        self.main_content.setColumnWidth(0, 60)   # Index
        self.main_content.setColumnWidth(1, 200)  # Clip name
        self.main_content.setColumnWidth(2, 100)  # Time
        
        # Set model column widths
        for i, model in enumerate(sorted_models):
            col_index = 3 + i
            self.main_content.setColumnWidth(col_index, 200)
        
        # Actions column
        actions_col = 3 + len(sorted_models)
        self.main_content.setColumnWidth(actions_col, 80)
        
        segment_index = 1
        total_segments = 0
        shown_segments = 0
        
        for clip_idx, result in enumerate(self.comparison_results):
            clip_name = result.clip_info.get("name", f"Clip {clip_idx + 1}")
            
            # Get all segments for this clip
            all_segments = self.combine_all_segments(result)
            segment_groups = self.group_segments_by_time(all_segments)
            
            # Initialize selections for this clip
            if clip_idx not in self.selected_transcriptions:
                self.selected_transcriptions[clip_idx] = {}
            
            for group_idx, segment_group in enumerate(segment_groups):
                total_segments += 1
                
                # CHECK IF WE SHOULD SHOW THIS SEGMENT GROUP
                if not self.should_show_segment_group(segment_group):
                    # Auto-select the base model for hidden identical segments
                    base_segment = next((seg for seg in segment_group if seg.get('is_base', False)), segment_group[0])
                    self.selected_transcriptions[clip_idx][group_idx] = base_segment
                    continue  # Skip showing this segment
                
                shown_segments += 1
                
                # Create main segment item (only for segments with differences)
                segment_item = QTreeWidgetItem(self.main_content)
                
                # Set basic information
                segment_item.setText(0, f"{segment_index:03d}")
                segment_item.setText(1, clip_name)
                
                # Time information
                start_time = min(seg['start'] for seg in segment_group)
                end_time = max(seg['end'] for seg in segment_group)
                segment_item.setText(2, f"{start_time:.1f}-{end_time:.1f}s")
                
                # Create a mapping of model to segment text
                model_segments = {}
                for seg in segment_group:
                    model_segments[seg['model']] = seg
                
                # Default to base model selection
                base_segment = next((seg for seg in segment_group if seg.get('is_base', False)), segment_group[0])
                self.selected_transcriptions[clip_idx][group_idx] = base_segment
                
                # Fill in model columns
                for i, model in enumerate(sorted_models):
                    col_index = 3 + i
                    if model in model_segments:
                        text = model_segments[model]['text']
                        segment_item.setText(col_index, text)
                        
                        # Make the cell editable
                        segment_item.setFlags(segment_item.flags() | Qt.ItemIsEditable)
                        
                        # Highlight selected model
                        if model_segments[model] == base_segment:
                            segment_item.setBackground(col_index, QColor(200, 255, 200))  # Light green
                            font = segment_item.font(col_index)
                            font.setBold(True)
                            segment_item.setFont(col_index, font)
                    else:
                        segment_item.setText(col_index, "")
                
                # Store data
                segment_item.setData(0, Qt.UserRole, {
                    "clip_idx": clip_idx,
                    "group_idx": group_idx,
                    "segments": segment_group,
                    "model_segments": model_segments,
                    "selected": base_segment,
                    "audio_file": result.clip_info.get("output_path")
                })
                
                # Add action button (delete)
                actions_col = 3 + len(sorted_models)
                delete_btn = QPushButton("✕")
                delete_btn.setFixedSize(25, 25)
                delete_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #e74c3c;
                        color: white;
                        border: none;
                        border-radius: 12px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #c0392b;
                    }
                """)
                delete_btn.clicked.connect(lambda checked, item=segment_item: self.delete_segment(item))
                self.main_content.setItemWidget(segment_item, actions_col, delete_btn)
                
                segment_index += 1
        
        # Update status with filtering info
        hidden_count = total_segments - shown_segments
        if hidden_count > 0:
            self.log(f"Showing {shown_segments}/{total_segments} segments ({hidden_count} identical segments hidden)")
        
        self.update_status_display()

    def should_show_segment_group(self, segment_group):
        """Determine if a segment group should be shown (has meaningful differences)"""
        if len(segment_group) <= 1:
            return True  # Always show if only one model produced this segment
        
        # Get all unique texts from this group
        texts = []
        for seg in segment_group:
            text = seg['text'].strip()
            if text:  # Only consider non-empty texts
                texts.append(text)
        
        if len(texts) <= 1:
            return False  # All empty or only one non-empty text
        
        # Check if all texts are essentially the same
        first_text = texts[0]
        for text in texts[1:]:
            if not self.are_texts_essentially_same(first_text, text):
                return True  # Found a meaningful difference
        
        return False  # All texts are essentially the same

    def are_texts_essentially_same(self, text1, text2):
        """Check if two texts are essentially the same"""
        if not text1 or not text2:
            return False
        
        # If texts are exactly the same, they're definitely the same
        if text1.strip() == text2.strip():
            return True
            
        import re
        
        def normalize_text(text):
            # Convert to lowercase
            normalized = text.lower().strip()
            
            # Handle common variations
            normalized = re.sub(r'\bOK\b', 'okay', normalized, flags=re.IGNORECASE)
            normalized = re.sub(r'\bok\b', 'okay', normalized, flags=re.IGNORECASE)
            
            # Normalize whitespace but preserve punctuation
            normalized = re.sub(r'\s+', ' ', normalized)
            
            # Remove only leading/trailing punctuation, keep internal punctuation
            normalized = re.sub(r'^[^\w]+|[^\w]+$', '', normalized)
            
            return normalized
        
        norm1 = normalize_text(text1)
        norm2 = normalize_text(text2)
        
        # Check exact match after normalization
        if norm1 == norm2:
            return True
        
        # Check very high similarity (98%+) only for very similar texts
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Only consider 98%+ similarity as "same" for very close matches
        if similarity >= 0.98:
            return True
        
        return False

    def update_statistics_display(self, overall_stats):
        """Update the statistics display"""
        stats_text = f"Overall Voting Statistics:\n"
        stats_text += f"Total Clips: {overall_stats['total_clips']}\n"
        stats_text += f"Total Words: {overall_stats['total_words']}\n"
        stats_text += f"Successfully Voted Words: {overall_stats['total_voted_words']}\n"
        
        if overall_stats['total_words'] > 0:
            success_rate = overall_stats['total_voted_words'] / overall_stats['total_words']
            stats_text += f"Overall Success Rate: {success_rate*100:.1f}%\n\n"
        
        stats_text += "Model Contributions:\n"
        sorted_models = sorted(overall_stats['model_contributions'].items(), 
                              key=lambda x: x[1], reverse=True)
        
        for model, count in sorted_models:
            if count > 0:
                percentage = count / overall_stats['total_voted_words'] * 100 if overall_stats['total_voted_words'] > 0 else 0
                stats_text += f"  {model}: {count} words ({percentage:.1f}%)\n"
        
        self.stats_text.setText(stats_text)
    
    def export_results(self):
        """Export voting results to file"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Voting Results", 
            "voting_results.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("WHISPER MODEL VOTING RESULTS\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, result in enumerate(self.voting_results):
                        f.write(f"CLIP {i+1}: {result.clip_info.get('name', 'Unknown')}\n")
                        f.write("-" * 40 + "\n")
                        
                        stats = result.voting_statistics
                        f.write(f"Voting Success Rate: {stats['voting_success_rate']*100:.1f}%\n")
                        f.write(f"Total Words: {stats['total_words']}\n")
                        f.write(f"Voted Words: {stats['voted_words']}\n\n")
                        
                        f.write("Model Contributions:\n")
                        for model, count in stats['model_agreement'].items():
                            if count > 0:
                                f.write(f"  {model}: {count} words\n")
                        f.write("\n")
                        
                        f.write("FINAL VOTED TRANSCRIPTION:\n")
                        if result.voted_segments:
                            full_text = ' '.join(segment.text for segment in result.voted_segments)
                            f.write(f"{full_text}\n")
                        f.write("\n" + "=" * 50 + "\n\n")
                
                QMessageBox.information(self, "Export Complete", f"Voting results exported to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")
    
    def create_voted_srt_files(self):
        """Create SRT files from voted transcriptions"""
        if not self.voting_results:
            QMessageBox.warning(self, "No Results", "No voting results available.")
            return
        
        try:
            output_dir = self.parent().output_dir if self.parent() else os.getcwd()
            created_files = []
            
            for i, result in enumerate(self.voting_results):
                clip_name = result.clip_info.get("name", f"Clip_{i+1}")
                
                if not result.voted_segments:
                    continue
                
                # Create SRT file
                safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in clip_name)
                srt_path = os.path.join(output_dir, f"{safe_name}_voted.srt")
                
                success = self.create_srt_from_voted_segments(result.voted_segments, srt_path)
                if success:
                    created_files.append(srt_path)
            
            if created_files:
                file_list = "\n".join(os.path.basename(f) for f in created_files)
                message = f"Created {len(created_files)} voted SRT files:\n\n{file_list}\n\n"
                message += "These files contain the word-level voted transcriptions from all models."
                
                QMessageBox.information(self, "SRT Files Created", message)
            else:
                QMessageBox.warning(self, "No Files Created", "No SRT files were created.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error Creating SRT Files", f"Error: {str(e)}")
    
    def create_srt_from_voted_segments(self, voted_segments, srt_path):
        """Create SRT file from voted segments"""
        try:
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(voted_segments, start=1):
                    if not segment.text or not segment.text.strip():
                        continue
                    
                    # Write segment number
                    f.write(f"{i}\n")
                    
                    # Write timestamps
                    start_time = self.parent().seconds_to_srt_time(segment.start)
                    end_time = self.parent().seconds_to_srt_time(segment.end)
                    f.write(f"{start_time} --> {end_time}\n")
                    
                    # Write text
                    f.write(f"{segment.text.strip()}\n\n")
            
            return True
        except Exception as e:
            print(f"Error creating SRT file: {str(e)}")
            return False

class SilenceDetector:
    """Class to detect silence segments in audio files"""
    def __init__(self):
        self.min_silence_duration = 0.2  # Minimum silence duration in seconds
        self.silence_threshold = -35  # Threshold in dB to consider as silence
        self.silence_padding = 0.05  # Padding around silence segments in seconds
        
    def log(self, message):
        """Log function for the SilenceDetector class"""
        print(f"[SilenceDetector] {message}")
        
    def run_ffmpeg_with_debug(self, cmd, operation_name, audio_file=None):
        # Format the command for logging
        cmd_str = " ".join(cmd)
        
        # Create a detailed log entry
        log_parts = [f"=== SilenceDetector - FFmpeg {operation_name} ==="]
        
        if audio_file:
            # Add audio file information to the log if available
            log_parts.append(f"Audio file: {audio_file}")
            
            # Add file size if available
            try:
                file_size = os.path.getsize(audio_file)
                log_parts.append(f"File size: {file_size/1024:.2f} KB")
            except:
                pass
        
        # Add the full command
        log_parts.append(f"Command: {cmd_str}")
        
        # Join all parts with newlines
        log_message = "\n".join(log_parts)
        self.log(log_message)
        
        # Start time for performance tracking
        start_time = time.time()
        
        # Execute the command
        try:
            process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            success = True
            elapsed = time.time() - start_time
            self.log(f"✓ FFmpeg {operation_name} completed successfully in {elapsed:.2f}s")
            return success, process.stdout, process.stderr
        except subprocess.CalledProcessError as e:
            success = False
            elapsed = time.time() - start_time
            self.log(f"✗ FFmpeg {operation_name} failed after {elapsed:.2f}s")
            self.log(f"Error: {e}")
            self.log(f"Error output: {e.stderr.decode() if hasattr(e, 'stderr') else 'No stderr'}")
            return success, e.stdout if hasattr(e, 'stdout') else None, e.stderr if hasattr(e, 'stderr') else None
        
    def detect_silences(self, audio_file):
        cmd = [
            "ffmpeg",
            "-i", audio_file,
            "-af", f"silencedetect=noise={self.silence_threshold}dB:d={self.min_silence_duration}",
            "-f", "null",
            "-"
        ]
        
        try:
            self.log(f"Detecting silence in {audio_file} with threshold={self.silence_threshold}dB, min_duration={self.min_silence_duration}s")
            success, stdout, stderr = self.run_ffmpeg_with_debug(cmd, "silence detection", audio_file)
            
            if not success:
                self.log(f"Silence detection failed for {audio_file}")
                return []
                
            stderr_output = stderr.decode()
            
            # Log raw output for debugging
            self.log(f"Raw silence detection output (first 500 chars): {stderr_output[:500]}")
            
            # Extract silence intervals using regex
            silence_starts = re.findall(r'silence_start: (\d+\.?\d*)', stderr_output)
            silence_ends = re.findall(r'silence_end: (\d+\.?\d*)', stderr_output)
            
            self.log(f"Found {len(silence_starts)} silence start points and {len(silence_ends)} silence end points")
            
            silences = []
            for i in range(len(silence_starts)):
                if i < len(silence_ends):
                    start_time = float(silence_starts[i])
                    end_time = float(silence_ends[i])
                    silences.append({
                        "start": max(0, start_time - self.silence_padding),
                        "end": end_time + self.silence_padding
                    })
            
            self.log(f"Created {len(silences)} silence segments with padding {self.silence_padding}s")
            
            # Log the first few silence segments for debugging
            if silences:
                for i, silence in enumerate(silences[:3]):  # Show first 3 silences
                    self.log(f"Silence {i}: {silence['start']:.2f}s - {silence['end']:.2f}s (duration: {silence['end'] - silence['start']:.2f}s)")
                if len(silences) > 3:
                    self.log(f"... and {len(silences) - 3} more silence segments")
            
            return silences
        except Exception as e:
            self.log(f"Error detecting silence: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return []
            
    def get_speech_segments(self, audio_file, total_duration):
        """Get speech segments (inverse of silence segments)
        
        Args:
            audio_file (str): Path to audio file
            total_duration (float): Total duration of the audio file
            
        Returns:
            list: List of dictionaries with speech start and end times
        """
        self.log(f"Finding speech segments in {audio_file} (total duration: {total_duration:.2f}s)")
        silences = self.detect_silences(audio_file)
        
        if not silences:
            # If no silences detected, return the entire clip as speech
            self.log(f"No silences detected, returning entire clip as speech: 0s - {total_duration:.2f}s")
            return [{"start": 0, "end": total_duration}]
            
        speech_segments = []
        
        # Add segment from start to first silence if needed
        if silences[0]["start"] > 0:
            speech_segments.append({"start": 0, "end": silences[0]["start"]})
            
        # Add segments between silences
        for i in range(len(silences) - 1):
            speech_segments.append({
                "start": silences[i]["end"],
                "end": silences[i + 1]["start"]
            })
            
        # Add segment from last silence to end if needed
        if silences[-1]["end"] < total_duration:
            speech_segments.append({"start": silences[-1]["end"], "end": total_duration})
            
        # Filter out very short segments (likely noise)
        min_speech_duration = 0.2  # Minimum speech segment duration in seconds
        speech_segments = [seg for seg in speech_segments if (seg["end"] - seg["start"]) >= min_speech_duration]
        
        self.log(f"Created {len(speech_segments)} speech segments after filtering (min duration: {min_speech_duration}s)")
        
        # Log the first few speech segments for debugging
        if speech_segments:
            for i, segment in enumerate(speech_segments[:3]):  # Show first 3 segments
                self.log(f"Speech {i}: {segment['start']:.2f}s - {segment['end']:.2f}s (duration: {segment['end'] - segment['start']:.2f}s)")
            if len(speech_segments) > 3:
                self.log(f"... and {len(speech_segments) - 3} more speech segments")
        
        return speech_segments

    def get_split_points(self, audio_file, total_duration):
        """Get split points for audio where silences are detected between speech segments
        
        Args:
            audio_file (str): Path to audio file
            total_duration (float): Total duration of the audio file
            
        Returns:
            list: List of time points to split the audio at (middle of silence segments)
        """
        self.log(f"Finding split points in {audio_file} (total duration: {total_duration:.2f}s)")
        silences = self.detect_silences(audio_file)
        speech_segments = self.get_speech_segments(audio_file, total_duration)
        
        # We need at least 2 speech segments to have a meaningful split
        if len(speech_segments) < 2:
            # Return empty list to indicate no split
            self.log(f"Less than 2 speech segments found ({len(speech_segments)}), no split needed")
            return []
        
        # Calculate middle points of silence segments
        split_points = []
        
        for i, silence in enumerate(silences):
            silence_start = silence["start"]
            silence_end = silence["end"]
            
            # Find the middle of the silence
            middle_point = (silence_start + silence_end) / 2
            
            # Only add split points that are actually between two speech segments
            # This avoids splitting at the beginning or end of the audio
            is_between_speech = False
            for j in range(len(speech_segments) - 1):
                if speech_segments[j]["end"] <= middle_point and speech_segments[j+1]["start"] >= middle_point:
                    is_between_speech = True
                    break
            
            if is_between_speech:
                split_points.append(middle_point)
                self.log(f"Added split point at {middle_point:.2f}s (middle of silence {i}: {silence_start:.2f}s - {silence_end:.2f}s)")
        
        self.log(f"Found {len(split_points)} split points in total")
        return split_points

class AudioClipProcessor(QThread):
    """Thread for processing audio clips."""
    progress_updated = pyqtSignal(int)
    clip_processed = pyqtSignal(dict)
    transcription_progress = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)  # Keep this for actual errors
    debug_info = pyqtSignal(str)      # Add this new signal for debug/info messages
    processing_finished = pyqtSignal(list)
    model_changed = pyqtSignal(str)
    
    def __init__(self, selected_clips, output_dir, max_workers=4, transcription_settings=None, vocal_isolation_settings=None, otio_output_dirs=None):
        super().__init__()
        self.selected_clips = selected_clips
        self.original_selected_clips = selected_clips.copy()
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.transcription_settings = transcription_settings or TranscriptionSettings()
        self.vocal_isolation_settings = vocal_isolation_settings
        self.transcriber = None
        self.silence_detector = SilenceDetector()
        self.processed_clips = []
        self.failed_normalizations = []
        
        # CRITICAL: Store OTIO output directories mapping
        self.otio_output_dirs = otio_output_dirs or {}
        
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Initialize enhanced transcriber if transcription is enabled
        if self.transcription_settings.enabled:
            self.transcriber = WhisperTranscriber(self.transcription_settings)

    def run_ffmpeg_with_debug(self, cmd, operation_name, clip_info=None):
        """
        Run ffmpeg command with detailed debugging output
        
        Args:
            cmd: List containing the ffmpeg command
            operation_name: String describing the operation
            clip_info: Optional dictionary with clip information
        
        Returns:
            Tuple of (success, stdout, stderr)
        """
        # Format the command for logging
        cmd_str = " ".join(cmd)
        
        # Create a detailed log entry
        log_parts = [f"=== FFmpeg {operation_name} ==="]
        
        if clip_info:
            # Add clip information to the log if available
            clip_name = clip_info.get("name", "Unknown")
            track_index = clip_info.get("track_index", "?")
            clip_index = clip_info.get("clip_index", "?")
            log_parts.append(f"Clip: {clip_name} (Track {track_index}, Clip {clip_index})")
            
            # Add timing information if available
            if "start_time" in clip_info and "duration" in clip_info:
                log_parts.append(f"Time: {clip_info['start_time']:.2f}s - {clip_info['start_time'] + clip_info['duration']:.2f}s (Duration: {clip_info['duration']:.2f}s)")
        
        # Add the full command
        log_parts.append(f"Command: {cmd_str}")
        
        # Join all parts with newlines
        log_message = "\n".join(log_parts)
        self.debug_info.emit(log_message)
        
        # Start time for performance tracking
        start_time = time.time()
        
        # Execute the command
        try:
            process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            success = True
            elapsed = time.time() - start_time
            print(f"✓ FFmpeg {operation_name} completed successfully in {elapsed:.2f}s")
            return success, process.stdout, process.stderr
        except subprocess.CalledProcessError as e:
            success = False
            elapsed = time.time() - start_time
            self.error_occurred.emit(f"✗ FFmpeg {operation_name} failed after {elapsed:.2f}s")
            self.error_occurred.emit(f"Error: {e}")
            self.error_occurred.emit(f"Error output: {e.stderr.decode() if hasattr(e, 'stderr') else 'No stderr'}")
            return success, e.stdout if hasattr(e, 'stdout') else None, e.stderr if hasattr(e, 'stderr') else None

    def run(self):
        """Enhanced run method with batch transcription (UPDATED)"""
        try:
            if not self.selected_clips:
                self.error_occurred.emit("No clips selected for processing")
                self.processing_finished.emit([])
                return
            
            clip_count = len(self.selected_clips)
            self.progress_updated.emit(0)
            
            # Step 1: Process all clips (extract audio, apply effects, etc.)
            self.debug_info.emit("=== STEP 1: Audio Processing ===")
            all_processed_clips = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for i, clip_data in enumerate(self.selected_clips, start=1):
                    # FIXED: Don't transcribe during individual processing
                    future = executor.submit(self.process_clip, clip_data, i, True)
                    futures[future] = i
                    
                completed = 0
                for future in futures:
                    try:
                        result = future.result()
                        if isinstance(result, list):
                            for segment_result in result:
                                all_processed_clips.append(segment_result)
                                self.clip_processed.emit(segment_result)
                        elif result:
                            all_processed_clips.append(result)
                            self.clip_processed.emit(result)
                    except Exception as e:
                        self.error_occurred.emit(f"Error processing clip: {str(e)}")
                        
                    completed += 1
                    progress = int(completed / clip_count * 50)  # First 50% for audio processing
                    self.progress_updated.emit(progress)
            
            # Sort clips by index
            all_processed_clips.sort(key=lambda x: x["index"])
            self.processed_clips = all_processed_clips
            
            # Step 2: Enhanced Batch Transcription
            if self.transcription_settings.enabled and all_processed_clips:
                self.debug_info.emit("=== STEP 2: Enhanced Batch Transcription ===")
                
                # Collect all audio paths for batch processing
                audio_paths = []
                clip_mapping = {}
                
                for i, clip_info in enumerate(all_processed_clips):
                    audio_path = clip_info.get("output_path")
                    if audio_path and os.path.exists(audio_path):
                        audio_paths.append(audio_path)
                        clip_mapping[audio_path] = i
                
                if audio_paths:
                    # FIXED: Initialize enhanced transcriber only once for batch processing
                    if not self.transcriber:
                        self.debug_info.emit("Initializing enhanced transcriber for batch processing")
                        self.transcriber = WhisperTranscriber(self.transcription_settings)
                    
                    # Batch transcribe all clips with the primary model
                    self.debug_info.emit(f"🚀 Batch transcribing {len(audio_paths)} clips with {self.transcription_settings.model_size}")
                    
                    batch_results = self.transcriber.batch_transcribe_clips(
                        audio_paths, 
                        model_size=self.transcription_settings.model_size,
                        timeout=self.transcription_settings.timeout_seconds
                    )
                    
                    # Process batch results
                    successful_transcriptions = 0
                    for audio_path, (segments, info, success) in zip(audio_paths, batch_results):
                        clip_index = clip_mapping[audio_path]
                        clip_info = all_processed_clips[clip_index]
                        
                        if success and segments:
                            # Adjust segments for padding
                            for segment in segments:
                                segment.start -= 1.0
                                segment.end -= 1.0
                                segment.start = max(0, segment.start)
                                segment.end = min(clip_info["duration"], segment.end)
                            
                            # Filter valid segments
                            valid_segments = [
                                s for s in segments if 
                                s.end > s.start and 
                                s.text and 
                                s.text.strip()
                            ]
                            
                            clip_info["segments"] = valid_segments
                            clip_info["transcription_info"] = info
                            
                            # Create SRT file
                            if self.transcription_settings.create_srt and valid_segments:
                                srt_path = os.path.splitext(audio_path)[0] + ".srt"
                                self.transcriber.create_srt(valid_segments, srt_path, clip_info["duration"])
                                clip_info["srt_path"] = srt_path
                            
                            successful_transcriptions += 1
                            
                            # Emit progress
                            progress = 50 + int((successful_transcriptions / len(audio_paths)) * 50)
                            self.progress_updated.emit(progress)
                        else:
                            display_name = clip_info.get('display_name', 'Unknown')
                            self.error_occurred.emit(f"Failed to transcribe: {display_name}")
                    
                    self.debug_info.emit(f"✓ Batch transcription completed: {successful_transcriptions}/{len(audio_paths)} successful")
            
            # Generate combined SRT if needed
            if self.transcription_settings.enabled and self.transcription_settings.create_combined_srt:
                self.create_combined_srt(self.processed_clips)
            
            self.processing_finished.emit(self.processed_clips)
            
        except Exception as e:
            self.error_occurred.emit(f"Error processing clips: {str(e)}")
            self.error_occurred.emit(traceback.format_exc())
            self.processing_finished.emit([])

    def process_clip(self, clip_data, index, apply_normalization=True):
        """Process a single audio clip with multiple segmentation options - FIXED for OTIO organization."""
        try:
            if not clip_data:
                self.error_occurred.emit(f"Invalid clip data for processing index {index}")
                return None
            
            # Extract index information for enhanced logging
            global_index = clip_data.get("global_index", index)
            track_index = clip_data.get("track_index", "?")
            clip_index = clip_data.get("clip_index", "?")
            file_path = clip_data.get("file_path", "")
            
            # CRITICAL: Get the correct output directory for this clip
            clip_output_dir = self.get_output_dir_for_clip(clip_data)
            
            # Debug logging
            self.debug_info.emit(f"=== Processing Cut #{global_index} ===")
            self.debug_info.emit(f"OTIO File: {os.path.basename(file_path)}")
            self.debug_info.emit(f"Output Directory: {clip_output_dir}")
            self.debug_info.emit(f"Track {track_index}, Clip {clip_index}")
            
            # Ensure the output directory exists
            os.makedirs(clip_output_dir, exist_ok=True)
            
            track_index_val = clip_data.get("track_index")
            clip_index_val = clip_data.get("clip_index")
            stream_index = clip_data.get("stream_index", 0)
            cut = clip_data.get("cut")
            
            if not track_index_val or not clip_index_val or not cut:
                self.error_occurred.emit(f"Missing required clip data for Cut #{global_index}")
                return None
            
            if not hasattr(cut, 'media_reference') or not cut.media_reference:
                self.error_occurred.emit(f"Cut #{global_index} has no media reference")
                return None
                    
            # Get media path from the reference
            if hasattr(cut.media_reference, 'target_url') and cut.media_reference.target_url:
                media_path = cut.media_reference.target_url
            else:
                self.error_occurred.emit(f"Cut #{global_index} has no valid media path")
                return None
                    
            if not os.path.exists(media_path):
                self.error_occurred.emit(f"Media file not found for Cut #{global_index}: {media_path}")
                return None
            
            # Get clip timing information
            start_time = cut.source_range.start_time.to_seconds()
            duration = cut.source_range.duration.to_seconds()
            
            # Store original timing information
            original_start = start_time
            original_duration = duration
            
            # Get clip name (for filename)
            clip_name = getattr(cut, 'name', f"Cut_{clip_index_val}")
            # Replace invalid filename characters
            clip_name = "".join([c if c.isalnum() or c in "._- " else "_" for c in clip_name])
            
            # Include global index in filename for easy identification
            if global_index != "?" and global_index != index:
                clip_name = f"Cut{global_index:03d}_{clip_name}"
            
            # Append suffix to indicate if normalization was applied
            normalization_suffix = "_normalized" if apply_normalization else "_raw"
            
            # Temporary path for extraction without padding
            temp_extract_path = os.path.join(self.temp_dir.name, f"temp_extract_{index}_{normalization_suffix}.wav")
            
            # Check the number of audio streams in the file
            audio_stream_count = get_audio_stream_count(media_path)
            
            # Safety check for the stream index
            if stream_index >= audio_stream_count:
                self.error_occurred.emit(f"Warning: Selected stream {stream_index} exceeds available streams ({audio_stream_count}) for Cut #{global_index}. Falling back to first stream.")
                stream_index = 0
            
            # Extract the audio segment without padding
            extract_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", media_path,
                "-t", str(duration),
                "-vn",  # No video
                "-map", f"0:a:{stream_index}",  # Use the actual stream index
                "-acodec", "pcm_s16le",  # Uncompressed audio
                "-ar", "48000",  # Sample rate
                "-ac", "2",      # Stereo
                temp_extract_path
            ]
            
            # Run FFmpeg for extraction with debug output
            clip_info = {
                "name": getattr(cut, 'name', f"Cut {clip_index_val}"),
                "global_index": global_index,
                "track_index": track_index_val,
                "stream_index": stream_index,
                "clip_index": clip_index_val,
                "start_time": start_time,
                "duration": duration
            }
            
            success, stdout, stderr = self.run_ffmpeg_with_debug(
                extract_cmd, 
                f"audio extraction for Cut #{global_index}", 
                clip_info
            )
            
            if not success:
                # Try fallback approaches...
                # (existing fallback code remains the same)
                if stream_index > 0:
                    self.error_occurred.emit(f"Failed with stream {stream_index} for Cut #{global_index}, trying with default stream (0)")
                    # Fallback to first stream
                    extract_cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(start_time),
                        "-i", media_path,
                        "-t", str(duration),
                        "-vn",  # No video
                        "-map", "0:a:0",  # Use the first audio stream
                        "-acodec", "pcm_s16le",
                        "-ar", "48000",
                        "-ac", "2",
                        temp_extract_path
                    ]
                    
                    success, stdout, stderr = self.run_ffmpeg_with_debug(
                        extract_cmd, 
                        f"audio extraction for Cut #{global_index} (fallback to first stream)", 
                        clip_info
                    )
                    
                    if not success:
                        # Final fallback without explicit stream mapping
                        extract_cmd = [
                            "ffmpeg", "-y",
                            "-ss", str(start_time),
                            "-i", media_path,
                            "-t", str(duration),
                            "-vn",  # No video
                            "-acodec", "pcm_s16le",
                            "-ar", "48000",
                            "-ac", "2",
                            temp_extract_path
                        ]
                        
                        success, stdout, stderr = self.run_ffmpeg_with_debug(
                            extract_cmd, 
                            f"audio extraction for Cut #{global_index} (without explicit mapping)", 
                            clip_info
                        )
                        
                        if not success:
                            error_msg = stderr.decode() if stderr else 'Unknown error'
                            self.error_occurred.emit(f"FFmpeg error extracting audio for Cut #{global_index}: {error_msg}")
                            return None
                else:
                    error_msg = stderr.decode() if stderr else 'Unknown error'
                    self.error_occurred.emit(f"FFmpeg error extracting audio for Cut #{global_index}: {error_msg}")
                    return None
            
            if not os.path.exists(temp_extract_path) or os.path.getsize(temp_extract_path) == 0:
                self.error_occurred.emit(f"Audio extraction failed for Cut #{global_index}, output file is empty or missing: {temp_extract_path}")
                return None
            
            # Get total duration of the extracted file with debug
            cmd = [
                "ffprobe", 
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                temp_extract_path
            ]
            
            self.debug_info.emit(f"=== FFprobe duration check for Cut #{global_index} ===\nCommand: {' '.join(cmd)}")
            
            try:
                duration_output = subprocess.check_output(cmd).decode().strip()
                extracted_duration = float(duration_output)
                self.debug_info.emit(f"Extracted duration for Cut #{global_index}: {extracted_duration}s")
            except Exception as e:
                self.error_occurred.emit(f"Error getting extracted audio duration for Cut #{global_index}: {str(e)}")
                extracted_duration = duration  # Fall back to original duration
            
            # NEW: Multiple segmentation options instead of silence detection
            split_points = []
            if not self.transcription_settings.fast_mode:
                self.debug_info.emit(f"Determining segmentation strategy for Cut #{global_index}...")
                split_points = self.get_segmentation_points(temp_extract_path, extracted_duration, global_index)
            
            # Process multiple segments or whole clip
            processed_clips = []
            
            if not split_points:
                # Process the entire clip as one segment
                self.debug_info.emit(f"No split points found in Cut #{global_index}, processing as single segment")
                
                # CRITICAL FIX: Use clip_output_dir instead of self.output_dir
                output_path = os.path.join(clip_output_dir, f"{clip_name}_track{track_index_val}_cut{clip_index_val}{normalization_suffix}.wav")
                
                self.debug_info.emit(f"Single segment output path: {output_path}")
                
                # Process single clip with full pipeline
                success = self.process_single_segment(
                    temp_extract_path, output_path, index, global_index, 
                    normalization_suffix, apply_normalization, clip_info
                )
                
                if not success:
                    return None
                
                # Add clip info with enhanced index information
                clip_info_result = {
                    "index": index,
                    "global_index": global_index,  # Global index for display
                    "display_name": f"Cut #{global_index}",  # Display name with index
                    "track_index": track_index_val,
                    "clip_index": clip_index_val,
                    "name": getattr(cut, 'name', f"Cut {clip_index_val}"),
                    "media_path": media_path,
                    "output_path": output_path,
                    "start_time": original_start,
                    "duration": original_duration,
                    "padding_start": 1.0,      # Fixed 1 second padding
                    "padding_end": 1.0,        # Fixed 1 second padding
                    "timeline_start": cut.range_in_parent().start_time.to_seconds(),
                    "timeline_end": cut.range_in_parent().start_time.to_seconds() + cut.range_in_parent().duration.to_seconds(),
                    "srt_path": None,
                    "segments": [],
                    "normalized": apply_normalization,
                    "file_path": file_path  # CRITICAL: Include file path for organization
                }
                
                processed_clips.append(clip_info_result)
                
            else:
                # Process multiple segments (split points found)
                self.debug_info.emit(f"Found {len(split_points)} split points in Cut #{global_index}")
                
                # Create segments based on split points
                segment_start_times = [0] + split_points
                segment_end_times = split_points + [extracted_duration]
                
                for seg_idx, (seg_start, seg_end) in enumerate(zip(segment_start_times, segment_end_times)):
                    # Skip segments that are too short
                    if seg_end - seg_start < 0.5:  # Skip segments shorter than 0.5 seconds
                        self.debug_info.emit(f"Skipping segment {seg_idx} of Cut #{global_index} (too short: {seg_end - seg_start:.2f}s)")
                        continue
                        
                    self.debug_info.emit(f"Processing segment {seg_idx} of Cut #{global_index} from {seg_start:.2f}s to {seg_end:.2f}s")
                    
                    # Extract this segment to a temp file
                    seg_extract_path = os.path.join(self.temp_dir.name, f"seg_{index}_{seg_idx}_{normalization_suffix}.wav")
                    
                    seg_extract_cmd = [
                        "ffmpeg", "-y",
                        "-i", temp_extract_path,
                        "-ss", str(seg_start),
                        "-t", str(seg_end - seg_start),
                        "-c:a", "pcm_s16le",
                        seg_extract_path
                    ]
                    
                    # Run FFmpeg for segment extraction with debug
                    segment_clip_info = {
                        "name": f"{getattr(cut, 'name', f'Cut {clip_index_val}')} (Segment {seg_idx+1})",
                        "global_index": f"{global_index}.{seg_idx+1}",  # Sub-index for segments
                        "track_index": track_index_val,
                        "clip_index": clip_index_val,
                        "segment_index": seg_idx,
                        "start_time": original_start + seg_start,
                        "duration": seg_end - seg_start
                    }
                    
                    seg_extract_success, seg_extract_stdout, seg_extract_stderr = self.run_ffmpeg_with_debug(
                        seg_extract_cmd, 
                        f"segment {seg_idx} extraction for Cut #{global_index}", 
                        segment_clip_info
                    )
                    
                    if not seg_extract_success:
                        self.error_occurred.emit(f"FFmpeg error extracting segment {seg_idx} of Cut #{global_index}: {seg_extract_stderr.decode() if seg_extract_stderr else 'Unknown error'}")
                        continue
                    
                    # CRITICAL FIX: Use clip_output_dir for segments too
                    segment_output_path = os.path.join(
                        clip_output_dir,  # Changed from self.output_dir
                        f"{clip_name}_track{track_index_val}_cut{clip_index_val}_seg{seg_idx}{normalization_suffix}.wav"
                    )
                    
                    self.debug_info.emit(f"Segment {seg_idx} output path: {segment_output_path}")
                    
                    # Process this segment with full pipeline
                    success = self.process_single_segment(
                        seg_extract_path, segment_output_path, f"{index}_{seg_idx}", 
                        f"{global_index}.{seg_idx+1}", normalization_suffix, 
                        apply_normalization, segment_clip_info
                    )
                    
                    if not success:
                        continue
                    
                    # Calculate actual start time in original media for this segment
                    segment_orig_start = original_start + seg_start
                    segment_orig_duration = seg_end - seg_start
                    
                    # Add clip info for this segment with enhanced indexing
                    segment_clip_info_result = {
                        "index": index * 100 + seg_idx,  # Create unique index for each segment
                        "global_index": f"{global_index}.{seg_idx+1}",  # Global index with segment number
                        "display_name": f"Cut #{global_index}.{seg_idx+1}",  # Display name with segment index
                        "track_index": track_index_val,
                        "clip_index": clip_index_val,
                        "segment_index": seg_idx,
                        "name": f"{getattr(cut, 'name', f'Cut {clip_index_val}')} (Segment {seg_idx+1})",
                        "media_path": media_path,
                        "output_path": segment_output_path,
                        "start_time": segment_orig_start,
                        "duration": segment_orig_duration,
                        "padding_start": 1.0,
                        "padding_end": 1.0,
                        "timeline_start": cut.range_in_parent().start_time.to_seconds() + seg_start,
                        "timeline_end": cut.range_in_parent().start_time.to_seconds() + seg_end,
                        "srt_path": None,
                        "segments": [],
                        "normalized": apply_normalization,
                        "file_path": file_path  # CRITICAL: Include file path for organization
                    }
                    
                    processed_clips.append(segment_clip_info_result)
            
            # Clean up main temp files
            try:
                if os.path.exists(temp_extract_path):
                    os.remove(temp_extract_path)
                    self.debug_info.emit(f"Removed main temp file for Cut #{global_index}: {temp_extract_path}")
            except Exception as e:
                self.error_occurred.emit(f"Error cleaning up main temp file for Cut #{global_index}: {str(e)}")
            
            # Apply vocal isolation if enabled
            if hasattr(self, 'vocal_isolation_settings') and self.vocal_isolation_settings and self.vocal_isolation_settings.enabled:
                for i, clip_info_item in enumerate(processed_clips):
                    output_path = clip_info_item["output_path"]
                    if os.path.exists(output_path):
                        # Create path for isolated audio
                        isolated_output_path = os.path.splitext(output_path)[0] + "_isolated.wav"
                        
                        # Isolate vocals using demucs
                        display_name = clip_info_item.get("display_name", f"Cut #{global_index}")
                        self.debug_info.emit(f"Isolating vocals for {display_name}...")
                        success = self.isolate_vocal_track(output_path, isolated_output_path, clip_info_item)
                        
                        if success:
                            # Update the clip info to use the isolated audio
                            clip_info_item["original_audio_path"] = output_path
                            clip_info_item["output_path"] = isolated_output_path
                            clip_info_item["isolated"] = True
                            self.debug_info.emit(f"Vocal isolation successful for {display_name}")
                        else:
                            self.error_occurred.emit(f"Vocal isolation failed for {display_name}, using original audio")
            
            # Return all processed clips instead of just the first one
            return processed_clips if processed_clips else None
            
        except Exception as e:
            error_clip_id = clip_data.get('global_index', clip_data.get('clip_index', '?'))
            self.error_occurred.emit(f"Error processing Cut #{error_clip_id}: {str(e)}")
            self.error_occurred.emit(traceback.format_exc())
            return None

    def get_segmentation_points(self, audio_file, total_duration, global_index):
        """
        ALWAYS attempt to find pauses and split clips into smaller phrases.
        This is specifically designed to detect speech pauses in short clips
        and create multiple SRT segments from individual phrases.
        """
        self.debug_info.emit(f"Finding speech pauses in Cut #{global_index}: {total_duration:.2f}s")
        
        # Always try to find splits, regardless of duration
        # Use multiple methods in order of preference
        
        # Method 1: Audio level analysis (most reliable for speech pauses)
        split_points = self.detect_speech_pauses_by_volume(audio_file, total_duration, global_index)
        
        # Method 2: If no volume-based splits found, try spectral analysis
        if not split_points:
            split_points = self.detect_speech_pauses_by_spectral_flux(audio_file, total_duration, global_index)
        
        # Method 3: If still no splits, try RMS energy analysis
        if not split_points:
            split_points = self.detect_speech_pauses_by_rms_dips(audio_file, total_duration, global_index)
        
        # Method 4: Last resort - if clip is long enough, force split by time
        if not split_points and total_duration > 6.0:
            split_points = self.force_time_based_splits(total_duration, global_index)
        
        if split_points:
            self.debug_info.emit(f"Cut #{global_index}: Found {len(split_points)} speech pauses for splitting")
            for i, point in enumerate(split_points):
                self.debug_info.emit(f"  Pause {i+1}: {point:.2f}s")
        else:
            self.debug_info.emit(f"Cut #{global_index}: No speech pauses detected - keeping as single phrase")
        
        return split_points

    def detect_speech_pauses_by_volume(self, audio_file, total_duration, global_index):
        """
        Detect speech pauses using silencedetect with specific duration targeting 0.2-0.3 seconds.
        Works for any clip duration.
        """
        try:
            self.debug_info.emit(f"Analyzing for 0.2-0.3 second speech pauses in Cut #{global_index}")
            
            # Target silence duration: 0.2-0.3 seconds
            target_silence_duration = 0.25  # 250ms - middle of target range
            
            # Use FFmpeg silencedetect with our target duration
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_file,
                "-af", f"silencedetect=noise=-35dB:d={target_silence_duration}",
                "-f", "null", "-"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            stderr_output = result.stderr
            
            # Parse silence periods
            silence_starts = []
            silence_ends = []
            
            for line in stderr_output.split('\n'):
                if 'silence_start:' in line:
                    try:
                        start_time = float(line.split('silence_start:')[1].split()[0])
                        silence_starts.append(start_time)
                    except (ValueError, IndexError):
                        continue
                elif 'silence_end:' in line:
                    try:
                        end_time = float(line.split('silence_end:')[1].split()[0])
                        silence_ends.append(end_time)
                    except (ValueError, IndexError):
                        continue
            
            self.debug_info.emit(f"Cut #{global_index}: Found {len(silence_starts)} silence start points, {len(silence_ends)} silence end points")
            
            if not silence_starts and not silence_ends:
                self.debug_info.emit(f"Cut #{global_index}: No 0.25s+ silence periods detected")
                return []
            
            # Convert silence periods to split points, filtering for our target duration
            return self.convert_silence_to_splits_filtered(silence_starts, silence_ends, total_duration, global_index)
            
        except subprocess.CalledProcessError as e:
            self.debug_info.emit(f"Silence detection failed for Cut #{global_index}: {str(e)}")
            return []
        except Exception as e:
            self.error_occurred.emit(f"Error in volume analysis for Cut #{global_index}: {str(e)}")
            return []

    def convert_silence_to_splits_filtered(self, silence_starts, silence_ends, total_duration, global_index):
        """
        Convert detected silence periods into split points, filtering for 0.2-0.3 second durations.
        """
        if not silence_starts and not silence_ends:
            return []
        
        # Create list of silence periods and filter by duration
        valid_silence_periods = []
        min_target_duration = 0.4
        
        # Match starts with ends
        for i in range(len(silence_starts)):
            start = silence_starts[i]
            end = None
            
            # Find corresponding end
            for j, end_time in enumerate(silence_ends):
                if end_time > start:
                    end = end_time
                    break
            
            if end is None:
                # Silence extends to end of clip - skip this
                continue
            
            silence_duration = end - start
            
            # Filter for our target duration range
            if min_target_duration <= silence_duration:
                valid_silence_periods.append({
                    'start': start,
                    'end': end,
                    'center': start + (silence_duration / 2),
                    'duration': silence_duration
                })
                self.debug_info.emit(f"  Valid silence: {start:.2f}s-{end:.2f}s (duration: {silence_duration:.3f}s)")
            else:
                self.debug_info.emit(f"  Skipped silence: {start:.2f}s-{end:.2f}s (duration: {silence_duration:.3f}s - outside 0.2-0.4s range)")
        
        if not valid_silence_periods:
            self.debug_info.emit(f"Cut #{global_index}: No silence periods in 0.2-0.4s range found")
            return []
        
        # Convert to split points, ensuring reasonable speech segment lengths
        split_points = []
        min_speech_length = 0.5  # Minimum 500ms speech segment (shorter for any duration clips)
        last_split = 0
        
        for silence in valid_silence_periods:
            split_point = silence['center']
            
            # Check if this creates reasonable speech segments
            speech_before = split_point - last_split
            speech_after = total_duration - split_point
            
            if speech_before >= min_speech_length and speech_after >= min_speech_length:
                split_points.append(split_point)
                last_split = split_point
                self.debug_info.emit(f"  Split point: {split_point:.2f}s (silence duration: {silence['duration']:.3f}s)")
            else:
                self.debug_info.emit(f"  Skipped split at {split_point:.2f}s (would create speech segments: {speech_before:.2f}s + {speech_after:.2f}s)")
        
        return split_points

    def detect_speech_pauses_by_spectral_flux(self, audio_file, total_duration, global_index):
        """
        Fallback method: Try different silence detection thresholds to find 0.2-0.3s pauses.
        """
        try:
            self.debug_info.emit(f"Trying multiple thresholds for 0.2-0.3s pauses in Cut #{global_index}")
            
            # Try multiple noise thresholds to find our target duration pauses
            thresholds = ['-30dB', '-40dB', '-45dB', '-50dB']
            target_duration = 0.25  # 250ms target
            
            for threshold in thresholds:
                self.debug_info.emit(f"  Trying threshold: {threshold}")
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", audio_file,
                    "-af", f"silencedetect=noise={threshold}:d={target_duration}",
                    "-f", "null", "-"
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    stderr_output = result.stderr
                    
                    # Parse silence periods
                    silence_starts = []
                    silence_ends = []
                    
                    for line in stderr_output.split('\n'):
                        if 'silence_start:' in line:
                            try:
                                start_time = float(line.split('silence_start:')[1].split()[0])
                                silence_starts.append(start_time)
                            except (ValueError, IndexError):
                                continue
                        elif 'silence_end:' in line:
                            try:
                                end_time = float(line.split('silence_end:')[1].split()[0])
                                silence_ends.append(end_time)
                            except (ValueError, IndexError):
                                continue
                    
                    # Check if we found any silence in our target range
                    if silence_starts:
                        self.debug_info.emit(f"  Found {len(silence_starts)} silence periods with {threshold}")
                        # Filter and return splits for target duration
                        return self.convert_silence_to_splits_filtered(silence_starts, silence_ends, total_duration, global_index)
                    
                except subprocess.CalledProcessError:
                    self.debug_info.emit(f"  {threshold} failed, trying next threshold")
                    continue
            
            self.debug_info.emit(f"Cut #{global_index}: No 0.2-0.3s pauses found with any threshold")
            return []
            
        except Exception as e:
            self.error_occurred.emit(f"Error in multi-threshold detection for Cut #{global_index}: {str(e)}")
            return []

    def detect_speech_pauses_by_rms_dips(self, audio_file, total_duration, global_index):
        """
        Final fallback: Try very sensitive silence detection for 0.2-0.3s pauses.
        """
        try:
            self.debug_info.emit(f"Trying very sensitive detection for 0.2-0.3s pauses in Cut #{global_index}")
            
            # Try with shorter minimum duration and more sensitive thresholds
            short_durations = [0.15, 0.2, 0.25]  # Try detecting even shorter pauses
            sensitive_thresholds = ['-25dB', '-35dB', '-45dB']
            
            for duration in short_durations:
                for threshold in sensitive_thresholds:
                    self.debug_info.emit(f"  Trying {threshold} with {duration}s minimum duration")
                    
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", audio_file,
                        "-af", f"silencedetect=noise={threshold}:d={duration}",
                        "-f", "null", "-"
                    ]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        stderr_output = result.stderr
                        
                        # Parse and filter for our target durations
                        silence_starts = []
                        silence_ends = []
                        
                        for line in stderr_output.split('\n'):
                            if 'silence_start:' in line:
                                try:
                                    start_time = float(line.split('silence_start:')[1].split()[0])
                                    silence_starts.append(start_time)
                                except (ValueError, IndexError):
                                    continue
                            elif 'silence_end:' in line:
                                try:
                                    end_time = float(line.split('silence_end:')[1].split()[0])
                                    silence_ends.append(end_time)
                                except (ValueError, IndexError):
                                    continue
                        
                        if silence_starts:
                            self.debug_info.emit(f"  Found silence with {threshold} @ {duration}s minimum")
                            # Use our filtering function to get 0.2-0.3s pauses
                            splits = self.convert_silence_to_splits_filtered(silence_starts, silence_ends, total_duration, global_index)
                            if splits:
                                return splits
                            
                    except subprocess.CalledProcessError:
                        continue
            
            self.debug_info.emit(f"Cut #{global_index}: No 0.2-0.3s pauses found with sensitive detection")
            return []
            
        except Exception as e:
            self.error_occurred.emit(f"Error in sensitive detection for Cut #{global_index}: {str(e)}")
            return []

    def force_time_based_splits(self, total_duration, global_index):
        """
        Last resort: force splits based on time for longer clips when no pauses are detected.
        """
        if total_duration < 4.0:
            return []
        
        # For clips longer than 6 seconds, create splits every 3-4 seconds
        split_interval = 3.5  # 3.5 second intervals
        split_points = []
        
        current_time = split_interval
        while current_time < total_duration - 1.5:  # Leave at least 1.5s for last segment
            split_points.append(current_time)
            current_time += split_interval
        
        if split_points:
            self.debug_info.emit(f"Cut #{global_index}: Forced time-based splits (no pauses detected)")
        
        return split_points

    def get_output_dir_for_clip(self, clip_data):
        """Get the appropriate output directory for a clip based on its OTIO source."""
        file_path = clip_data.get("file_path")
        
        # Debug logging
        self.debug_info.emit(f"Getting output dir for clip from file: {os.path.basename(file_path) if file_path else 'NO FILE PATH'}")
        
        if hasattr(self, 'otio_output_dirs') and file_path and file_path in self.otio_output_dirs:
            output_dir = self.otio_output_dirs[file_path]
            self.debug_info.emit(f"Found OTIO-specific directory: {output_dir}")
            return output_dir
        else:
            # Fallback to base directory
            self.debug_info.emit(f"Using fallback directory: {self.output_dir}")
            self.debug_info.emit(f"Available OTIO dirs: {list(self.otio_output_dirs.keys()) if hasattr(self, 'otio_output_dirs') else 'None'}")
            return self.output_dir

    def process_single_segment(self, input_path, output_path, segment_id, global_index, normalization_suffix, apply_normalization, clip_info):
        """Process a single audio segment with padding and normalization - UPDATED for OTIO organization."""
        try:
            self.debug_info.emit(f"Processing single segment: {global_index}")
            
            # Create 1-second silence files for padding
            silence_start_path = os.path.join(self.temp_dir.name, f"silence_start_{segment_id}_{normalization_suffix}.wav")
            silence_end_path = os.path.join(self.temp_dir.name, f"silence_end_{segment_id}_{normalization_suffix}.wav")
            
            # Generate 1 second of silence with same audio properties
            silence_cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-t", "1",  # 1 second of silence
                "-acodec", "pcm_s16le",
                silence_start_path
            ]
            
            # Run FFmpeg for silence generation with debug
            silence_success, silence_stdout, silence_stderr = self.run_ffmpeg_with_debug(
                silence_cmd, 
                f"silence generation for {global_index}", 
                clip_info
            )
            
            if not silence_success:
                error_msg = silence_stderr.decode() if silence_stderr else 'Unknown error'
                self.error_occurred.emit(f"Error creating silence files for {global_index}: {error_msg}")
                return False
            
            try:
                # Copy the silence file for end padding too
                self.debug_info.emit(f"Copying silence file from {silence_start_path} to {silence_end_path}")
                shutil.copy(silence_start_path, silence_end_path)
            except (subprocess.CalledProcessError, OSError) as e:
                self.error_occurred.emit(f"Error creating silence files for {global_index}: {str(e)}")
                return False
            
            # Create a file list for concatenation
            concat_file = os.path.join(self.temp_dir.name, f"concat_{segment_id}_{normalization_suffix}.txt")
            with open(concat_file, 'w') as f:
                f.write(f"file '{silence_start_path}'\n")
                f.write(f"file '{input_path}'\n")
                f.write(f"file '{silence_end_path}'\n")
            
            self.debug_info.emit(f"Created concatenation file for {global_index}: {concat_file}")
            
            # Concatenate files to create padded audio
            concat_path = os.path.join(self.temp_dir.name, f"concat_output_{segment_id}_{normalization_suffix}.wav")
            
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                concat_path
            ]
            
            # Run FFmpeg for concatenation with debug
            concat_success, concat_stdout, concat_stderr = self.run_ffmpeg_with_debug(
                concat_cmd, 
                f"audio concatenation for {global_index}", 
                clip_info
            )
            
            if not concat_success:
                error_msg = concat_stderr.decode() if concat_stderr else 'Unknown error'
                self.error_occurred.emit(f"FFmpeg error concatenating audio for {global_index}: {error_msg}")
                return False
            
            if not os.path.exists(concat_path) or os.path.getsize(concat_path) == 0:
                self.error_occurred.emit(f"Concatenated audio file is empty or missing for {global_index}: {concat_path}")
                return False
                
            # Apply audio normalization if requested
            if apply_normalization:
                # Use a less aggressive normalization approach
                normalize_cmd = [
                    "ffmpeg", "-y",
                    "-i", concat_path,
                    "-af", "loudnorm=I=-16:LRA=11:TP=-1.5:linear=true",  # EBU R128 loudness normalization with linear mode
                    "-ar", "48000",
                    "-acodec", "pcm_s16le",
                    output_path
                ]
                
                # Run FFmpeg for normalization with debug
                normalize_success, normalize_stdout, normalize_stderr = self.run_ffmpeg_with_debug(
                    normalize_cmd, 
                    f"audio normalization for {global_index}", 
                    clip_info
                )
                
                if not normalize_success:
                    error_msg = normalize_stderr.decode() if normalize_stderr else 'Unknown error'
                    self.error_occurred.emit(f"FFmpeg error normalizing audio for {global_index}: {error_msg}")
                    return False
            else:
                # Just copy the concatenated file without normalization
                copy_cmd = [
                    "ffmpeg", "-y",
                    "-i", concat_path,
                    "-ar", "48000",
                    "-acodec", "pcm_s16le",
                    output_path
                ]
                
                # Run FFmpeg for copying with debug
                copy_success, copy_stdout, copy_stderr = self.run_ffmpeg_with_debug(
                    copy_cmd, 
                    f"audio copying (no normalization) for {global_index}", 
                    clip_info
                )
                
                if not copy_success:
                    error_msg = copy_stderr.decode() if copy_stderr else 'Unknown error'
                    self.error_occurred.emit(f"FFmpeg error copying audio for {global_index}: {error_msg}")
                    return False
            
            # Clean up temp files
            try:
                for file in [silence_start_path, silence_end_path, concat_file, concat_path]:
                    if os.path.exists(file):
                        os.remove(file)
                        self.debug_info.emit(f"Removed temp file: {file}")
            except Exception as e:
                self.error_occurred.emit(f"Error cleaning up temp files for {global_index}: {str(e)}")
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error processing single segment {global_index}: {str(e)}")
            return False

    def transcribe_segment(self, clip_info):
        """Enhanced transcribe segment method - SINGLE CLIP ONLY (FIXED)"""
        try:
            clip_index = clip_info.get("clip_index")
            global_index = clip_info.get("global_index", "?")
            output_path = clip_info.get("output_path")
            display_name = clip_info.get("display_name", f"Cut #{global_index}")
            
            self.transcription_progress.emit({
                "clip_index": clip_index,
                "global_index": global_index,
                "status": "starting",
                "progress": 0
            })
            
            # FIXED: Check if transcriber exists and initialize if needed
            if not self.transcriber:
                self.debug_info.emit(f"Initializing transcriber for {display_name}")
                try:
                    self.transcriber = WhisperTranscriber(self.transcription_settings)
                except Exception as e:
                    self.error_occurred.emit(f"Failed to initialize transcriber for {display_name}: {str(e)}")
                    return False
            
            # FIXED: Check if model is loaded
            if not self.transcriber.model:
                self.debug_info.emit(f"Loading model for {display_name}")
                try:
                    self.transcriber.load_model()
                except Exception as e:
                    self.error_occurred.emit(f"Failed to load model for {display_name}: {str(e)}")
                    return False
            
            # FIXED: Validate audio file exists and is accessible
            if not output_path or not os.path.exists(output_path):
                self.error_occurred.emit(f"Audio file not found for {display_name}: {output_path}")
                return False
            
            # FIXED: Check file size
            try:
                file_size = os.path.getsize(output_path)
                if file_size < 1024:  # Less than 1KB
                    self.error_occurred.emit(f"Audio file too small for {display_name}: {file_size} bytes")
                    return False
            except Exception as e:
                self.error_occurred.emit(f"Cannot access audio file for {display_name}: {str(e)}")
                return False
            
            # Use safe transcription with retries
            self.debug_info.emit(f"Starting transcription for {display_name}")
            segments, result, success = self.transcriber.safe_transcribe_single(output_path, max_retries=2)
            
            if success and segments:
                # Adjust segment timestamps to account for padding
                for segment in segments:
                    segment.start -= 1.0  # Remove padding
                    segment.end -= 1.0
                    segment.start = max(0, segment.start)
                    segment.end = min(clip_info["duration"], segment.end)
                
                # Filter out empty segments
                valid_segments = [
                    s for s in segments if 
                    s.end > s.start and 
                    s.text and 
                    s.text.strip() and 
                    "transcript of speech audio" not in s.text.lower()
                ]
                
                clip_info["segments"] = valid_segments
                clip_info["transcription_info"] = result
                
                # Create SRT file for this clip if needed
                if self.transcription_settings.create_srt and valid_segments:
                    srt_path = os.path.splitext(output_path)[0] + ".srt"
                    success = self.transcriber.create_srt(valid_segments, srt_path, clip_info["duration"])
                    if success:
                        clip_info["srt_path"] = srt_path
                        self.debug_info.emit(f"Created SRT for {display_name}: {os.path.basename(srt_path)}")
                
                self.transcription_progress.emit({
                    "clip_index": clip_index,
                    "global_index": global_index,
                    "status": "completed",
                    "progress": 100
                })
                
                self.debug_info.emit(f"✓ Transcription completed for {display_name}: {len(valid_segments)} segments")
                return True
            else:
                self.error_occurred.emit(f"No valid transcription for {display_name}")
                return False
                    
        except Exception as e:
            display_name = clip_info.get("display_name", f"Cut #{clip_info.get('global_index', '?')}")
            self.error_occurred.emit(f"Transcription error for {display_name}: {str(e)}")
            import traceback
            self.debug_info.emit(traceback.format_exc())
            return False

    def create_srt(self, segments, srt_path, original_duration=None):
        """Create an SRT subtitle file from transcription segments with optional fixed end timestamp
        
        Args:
            segments: List of transcription segments
            srt_path: Path to save the SRT file
            original_duration: If provided, the last segment's end timestamp will be set to this value
        """
        try:
            if not segments:
                print(f"No segments to create SRT for {srt_path}")
                return False
                
            with open(srt_path, 'w', encoding='utf-8') as f:
                # Sort segments by start time to ensure proper ordering
                sorted_segments = sorted(segments, key=lambda x: x.start)
                
                # For all segments except the last one, use their original end timestamps
                for i, segment in enumerate(sorted_segments[:-1], start=1):
                    # Skip segments with empty text
                    if not segment.text or not segment.text.strip():
                        continue
                        
                    # Write segment number
                    f.write(f"{i}\n")
                    
                    # Write timestamps in SRT format (00:00:00,000 --> 00:00:00,000)
                    start_time = seconds_to_srt_time(segment.start)
                    end_time = seconds_to_srt_time(segment.end)
                    f.write(f"{start_time} --> {end_time}\n")
                    
                    # Write text content
                    f.write(f"{segment.text.strip()}\n\n")
                
                # Handle the last segment with special end timestamp if original_duration is provided
                if len(sorted_segments) > 0:
                    last_segment = sorted_segments[-1]
                    
                    # Skip if empty text
                    if last_segment.text and last_segment.text.strip():
                        # Write segment number
                        segment_number = sum(1 for s in sorted_segments[:-1] if s.text and s.text.strip()) + 1
                        f.write(f"{segment_number}\n")
                        
                        # Use original start time but set end time based on original_duration if provided
                        start_time = seconds_to_srt_time(last_segment.start)
                        
                        if original_duration is not None:
                            # Use original duration as end timestamp
                            end_time = seconds_to_srt_time(original_duration)
                        else:
                            # Otherwise use the segment's computed end time
                            end_time = seconds_to_srt_time(last_segment.end)
                            
                        f.write(f"{start_time} --> {end_time}\n")
                        
                        # Write text content
                        f.write(f"{last_segment.text.strip()}\n\n")
                
            print(f"Created SRT file: {srt_path}")
            return True
        except Exception as e:
            print(f"Error creating SRT file: {str(e)}")
            print(traceback.format_exc())
            return False

    def parse_srt_time(self, time_str):
        """Parse SRT time format to seconds"""
        parts = time_str.strip().split(',')
        if len(parts) != 2:
            return 0
        
        time_parts = parts[0].split(':')
        if len(time_parts) != 3:
            return 0
        
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = int(time_parts[2])
        milliseconds = int(parts[1])
        
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    def log(self, message, error=False):
        """Log function for the AudioClipProcessor class"""
        # Forward log messages to the error_occurred signal
        # This will be displayed in the UI via the OTIOAudioEditor's log method
        self.error_occurred.emit(message)

    def create_combined_srt(self, processed_clips):
        """Create a combined SRT file from all processed clips, using timeline timestamps."""
        if not processed_clips:
            self.log("No clips processed, can't create combined SRT.")
            return None
                
        try:
            # Create a flattened list of clips
            timeline_clips = []
            for clip in processed_clips:
                # Handle both individual clips and lists of clips
                if isinstance(clip, list):
                    timeline_clips.extend(clip)
                else:
                    timeline_clips.append(clip)
                    
            # Now sort the flattened list
            timeline_clips = sorted(timeline_clips, key=lambda x: x.get("timeline_start", 0))
            
            # Path for the combined SRT
            combined_srt_path = os.path.join(self.output_dir, "combined_transcription.srt")
            
            # Create a clean combined SRT file
            # First, collect and sort all segments by timeline position
            all_segments = []
            
            # Debug - print how many clips we're processing
            self.log(f"Processing {len(timeline_clips)} clips for combined SRT")
            
            # Collect SRT files from all segments
            for clip in timeline_clips:
                srt_path = clip.get("srt_path")
                if not srt_path or not os.path.exists(srt_path):
                    continue
                    
                # Log which SRT we're parsing
                self.log(f"Parsing SRT file: {os.path.basename(srt_path)}")
                    
                # Get timeline offset for this clip
                timeline_start = clip.get("timeline_start", 0)
                
                # Parse the SRT file
                segments_from_srt = []
                try:
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Very simple SRT parser
                    blocks = content.strip().split('\n\n')
                    for block in blocks:
                        lines = block.split('\n')
                        if len(lines) >= 3:  # Need at least 3 lines (number, times, text)
                            # Parse time stamps
                            time_line = lines[1]
                            times = time_line.split(' --> ')
                            if len(times) == 2:
                                start_time = self.parse_srt_time(times[0])
                                end_time = self.parse_srt_time(times[1])
                                
                                # Get text (may be multiple lines)
                                text = '\n'.join(lines[2:])
                                
                                segments_from_srt.append({
                                    "start": start_time,
                                    "end": end_time,
                                    "text": text
                                })
                    
                    # Add these segments to our collection with timeline offset
                    for segment in segments_from_srt:
                        # Adjust for timeline position
                        all_segments.append({
                            "start": timeline_start + segment["start"],
                            "end": timeline_start + segment["end"],
                            "text": segment["text"]
                        })
                        
                    self.log(f"Parsed {len(segments_from_srt)} segments from {os.path.basename(srt_path)}")
                        
                except Exception as e:
                    self.log(f"Error parsing SRT file {srt_path}: {str(e)}")
            
            # Sort by start time
            all_segments.sort(key=lambda x: x["start"])
            
            # Check for and fix overlaps
            cleaned_segments = []
            
            for i, segment in enumerate(all_segments):
                # Skip empty segments
                if not segment["text"]:
                    continue
                
                # Start with the current segment
                clean_segment = segment.copy()
                
                # Check for overlap with the previous segment
                if cleaned_segments and clean_segment["start"] < cleaned_segments[-1]["end"]:
                    # Decide how to handle the overlap based on which text is longer/more important
                    if len(clean_segment["text"]) > len(cleaned_segments[-1]["text"]):
                        # Current segment is more important, adjust previous segment
                        cleaned_segments[-1]["end"] = clean_segment["start"] - 0.01
                    else:
                        # Previous segment is more important, adjust current segment
                        clean_segment["start"] = cleaned_segments[-1]["end"] + 0.01
                
                # Only add if the segment has a positive duration after adjustments
                if clean_segment["end"] > clean_segment["start"]:
                    cleaned_segments.append(clean_segment)
            
            # Format segments with proper casing and punctuation
            for i, segment in enumerate(cleaned_segments):
                text = segment["text"].strip()
                
                # Apply formatting rules based on position
                if i == 0:  # First segment
                    # Capitalize first letter
                    if text:
                        text = text[0].upper() + text[1:]
                    # Remove any trailing period if present
                    if text and text.endswith("."):
                        text = text[:-1]
                elif i == len(cleaned_segments) - 1:  # Last segment
                    # Ensure lowercase first letter for non-first segment
                    if text:
                        text = text[0].lower() + text[1:]
                    # Ensure period at end
                    if text and not text.endswith("."):
                        text = text + "."
                else:  # Middle segments
                    # Ensure lowercase first letter
                    if text:
                        text = text[0].lower() + text[1:]
                    # Remove any trailing period if present
                    if text and text.endswith("."):
                        text = text[:-1]
                    
                # Update the segment text with formatted version
                segment["text"] = text
            
            # Write the cleaned segments to SRT
            segment_count = 0
            with open(combined_srt_path, 'w', encoding='utf-8') as srt_file:
                for segment in cleaned_segments:
                    segment_count += 1
                    
                    # Write segment number
                    srt_file.write(f"{segment_count}\n")
                    
                    # Write timestamps in SRT format
                    start_time_str = seconds_to_srt_time(segment["start"])
                    end_time_str = seconds_to_srt_time(segment["end"])
                    srt_file.write(f"{start_time_str} --> {end_time_str}\n")
                    
                    # Write text - limit to 2 lines max for gaming subtitles
                    text = segment["text"]
                    # Split on sentence boundaries if possible
                    if len(text) > 60:  # If text is long
                        sentences = text.split('. ')
                        if len(sentences) > 1:
                            formatted_text = '.\n'.join(sentences[:-1]) + '.\n' + sentences[-1]
                        else:
                            # If no sentence boundaries, split at a reasonable point
                            mid_point = len(text) // 2
                            # Find the nearest space
                            split_point = text.rfind(' ', 0, mid_point) if mid_point < len(text) else len(text)
                            if split_point > 0:
                                formatted_text = text[:split_point] + '\n' + text[split_point+1:]
                            else:
                                formatted_text = text
                    else:
                        formatted_text = text
                    
                    srt_file.write(f"{formatted_text}\n\n")
            
            if segment_count > 0:
                self.log(f"Created combined SRT with {segment_count} segments at {combined_srt_path}")
                return combined_srt_path
            else:
                self.log("No transcription segments found in clips, combined SRT not created.")
                return None
                
        except Exception as e:
            self.log(f"Error creating combined SRT: {str(e)}")
            self.log(traceback.format_exc())
            return None

    def isolate_vocal_track(self, audio_path, output_path, clip_info):
        """Isolate vocals from an audio file using Demucs"""
        try:
            if not hasattr(self, 'vocal_isolation_settings') or not self.vocal_isolation_settings:
                self.error_occurred.emit(f"Vocal isolation settings not configured")
                return False
                
            model = self.vocal_isolation_settings.model
            
            self.error_occurred.emit(f"Starting vocal isolation for {os.path.basename(audio_path)}...")
            
            # Create temp directory for demucs output
            demucs_output_dir = os.path.join(self.temp_dir.name, "demucs_output")
            os.makedirs(demucs_output_dir, exist_ok=True)
            
            # Environment variables for the subprocess
            env = os.environ.copy()
            env["NUMBA_DISABLE_INTEL_SVML"] = "1"
            env["MKL_THREADING_LAYER"] = "GNU"
            env["PYTHONIOENCODING"] = "utf-8"
            
            # Determine if we're extracting vocals, instruments, or both
            stems_arg = "--two-stems=vocals"
            
            # Run demucs to extract vocals
            command = [
                "demucs",
                stems_arg,
                "-o", demucs_output_dir,
                "-n", model,
                audio_path
            ]
            
            self.error_occurred.emit(f"Running command: {' '.join(command)}")
            
            # Run command with detailed debugging
            success, stdout, stderr = self.run_ffmpeg_with_debug(
                command, 
                "vocal isolation", 
                clip_info
            )
            
            if success:
                # Find the output vocals file
                track_name = os.path.splitext(os.path.basename(audio_path))[0]
                model_output_dir = os.path.join(demucs_output_dir, model)
                vocals_path = os.path.join(model_output_dir, track_name, "vocals.wav")
                
                if os.path.exists(vocals_path):
                    # Copy the vocals file to the output path
                    self.error_occurred.emit(f"Copying isolated vocals to: {output_path}")
                    shutil.copy(vocals_path, output_path)
                    return True
                else:
                    self.error_occurred.emit(f"Error: Vocals file not found in expected location")
                    return False
            else:
                self.error_occurred.emit(f"Error during vocal isolation: {stderr.decode() if stderr else 'Unknown error'}")
                return False
                
        except Exception as e:
            self.error_occurred.emit(f"Error during vocal isolation: {str(e)}")
            self.error_occurred.emit(traceback.format_exc())
            return False

class WhisperTranscriber:
    """Enhanced WhisperTranscriber that uses the new batch processing functions"""
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.current_model_size = settings.model_size
        
        # Add local WhisperX path at the beginning of sys.path
        local_whisperx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisperX")
        if local_whisperx_path not in sys.path:
            sys.path.insert(0, local_whisperx_path)
        
        # Import the enhanced whisperx
        try:
            import whisperx
            self.whisperx = whisperx
            print(f"✓ Using enhanced WhisperX from: {local_whisperx_path}")
        except ImportError as e:
            print(f"❌ Failed to import enhanced WhisperX: {e}")
            raise
        
        # Track failed clips for later retry
        self.failed_clips = []
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            try:
                free_memory, total_memory = torch.cuda.mem_get_info()
                free_memory_gb = free_memory / (1024**3)
                total_memory_gb = total_memory / (1024**3)
                print(f"GPU Memory: {free_memory_gb:.2f}GB free of {total_memory_gb:.2f}GB total")
            except:
                print("CUDA is available but memory info couldn't be retrieved")
        else:
            print("CUDA is not available, using CPU")

    def load_model(self, model_size=None):
        """Load model using enhanced WhisperX with compatibility fixes"""
        model_size = model_size or self.current_model_size
        
        print(f"Loading enhanced WhisperX model: {model_size}")
        
        try:
            # Create compatible ASR options without problematic parameters
            asr_options = {
                "beam_size": getattr(self.settings, 'beam_size', 5),
                "best_of": 5,
                "patience": 1,
                "length_penalty": 1,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.3,  # More sensitive to detect quiet speech
                "condition_on_previous_text": False,
                "suppress_tokens": [-1],
                "without_timestamps": False,
            }
            
            # Try enhanced loading first (without problematic VAD parameters)
            try:
                self.model = self.whisperx.load_model(
                    model_size,
                    device=self.settings.device,
                    compute_type=getattr(self.settings, 'compute_type', 'int8'),
                    language=self.settings.language if self.settings.language != "auto" else None,
                    asr_options=asr_options,
                    # Remove VAD parameters that cause compatibility issues
                )
                print(f"✓ Successfully loaded enhanced model without VAD: {model_size}")
                
            except Exception as vad_error:
                print(f"Enhanced loading failed: {str(vad_error)}")
                
                # Fallback to minimal parameter loading
                print("🔄 Trying minimal parameter model loading...")
                self.model = self.whisperx.load_model(
                    model_size,
                    device=self.settings.device,
                    compute_type=getattr(self.settings, 'compute_type', 'int8'),
                    language=self.settings.language if self.settings.language != "auto" else None
                )
                print(f"✓ Successfully loaded minimal model: {model_size}")
            
            self.current_model_size = model_size
            return self.model
            
        except Exception as e:
            print(f"❌ All model loading attempts failed for {model_size}: {str(e)}")
            
            # Final fallback: try loading without any optional parameters
            try:
                print("🔄 Trying absolute minimal model loading...")
                self.model = self.whisperx.load_model(model_size, device=self.settings.device)
                self.current_model_size = model_size
                print(f"✓ Successfully loaded absolute minimal model: {model_size}")
                return self.model
            except Exception as final_error:
                print(f"❌ Final fallback also failed: {str(final_error)}")
                raise

    def batch_transcribe_clips(self, audio_paths, model_size=None, timeout=None):
        """Improved batch transcription with better error handling"""
        model = self.load_model(model_size)
        if not model:
            return [([], {}, False) for _ in audio_paths]
        
        print(f"🚀 Improved batch transcription of {len(audio_paths)} clips with model {self.current_model_size}")
        
        try:
            # Check if enhanced batch method is available
            if hasattr(model, 'transcribe_multiple_clips'):
                print("Using enhanced batch processing")
                try:
                    results = model.transcribe_multiple_clips(
                        audio_paths,
                       # max_parallel_clips=getattr(self.settings, 'batch_size', 8),  # Reduced for stability
                        batch_size=1,
                        language=self.settings.language if self.settings.language != "auto" else None,
                        task="transcribe",
                        chunk_size=30,
                        print_progress=True,
                        verbose=False,
                    )
                except Exception as batch_error:
                    print(f"Enhanced batch processing failed: {batch_error}")
                    raise  # Fall through to individual processing
            else:
                print("Enhanced batch method not available, using individual processing")
                raise AttributeError("No batch method")
            
        except (AttributeError, Exception):
            print("⚠️ Batch processing unavailable, using improved individual processing...")
            results = []
            
            for i, audio_path in enumerate(audio_paths):
                print(f"Processing file {i+1}/{len(audio_paths)}: {os.path.basename(audio_path)}")
                
                try:
                    segments, info, success = self.safe_transcribe_single(audio_path)
                    
                    if success:
                        # Convert segments to expected format for batch results
                        batch_result = {
                            "segments": [],
                            "language": info.get("language", "en")
                        }
                        
                        for seg in segments:
                            batch_result["segments"].append({
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text
                            })
                        
                        results.append(batch_result)
                        print(f"  ✓ Success: {len(segments)} segments")
                    else:
                        results.append({"segments": [], "language": "en"})
                        skip_reason = info.get("skip_reason", "unknown")
                        print(f"  ✗ Failed: {skip_reason}")
                        
                except Exception as e:
                    print(f"  ❌ Individual processing failed: {str(e)}")
                    results.append({"segments": [], "language": "en"})
                
                # Periodic memory cleanup
                if i % 10 == 0 and i > 0:
                    print(f"Memory cleanup after {i} files")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
        
        # Convert results to expected format
        processed_results = []
        successful_count = 0
        
        for i, result in enumerate(results):
            if result and result.get("segments"):
                # Convert segments to expected format
                segments = []
                for seg in result["segments"]:
                    class Segment:
                        def __init__(self, start, end, text):
                            self.start = start
                            self.end = end
                            self.text = text
                    
                    segments.append(Segment(seg["start"], seg["end"], seg["text"]))
                
                processed_results.append((segments, result, True))
                successful_count += 1
            else:
                processed_results.append(([], {}, False))
        
        print(f"✓ Improved batch transcription completed: {successful_count}/{len(audio_paths)} successful")
        
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return processed_results

    def safe_transcribe_single(self, audio_path, max_retries=3):
        """Safely transcribe a single audio file with retries"""
        for attempt in range(max_retries):
            try:
                if not self.model:
                    self.load_model()
                
                result = self.model.transcribe(
                    audio_path,
                    batch_size=8,
                    language=self.settings.language if self.settings.language != "auto" else None,
                    task="transcribe"
                )
                
                if result and result.get("segments"):
                    segments = []
                    for seg in result["segments"]:
                        class Segment:
                            def __init__(self, start, end, text):
                                self.start = start
                                self.end = end
                                self.text = text
                        
                        segments.append(Segment(seg["start"], seg["end"], seg["text"]))
                    
                    return segments, result, True
                else:
                    print(f"⚠️ No segments found for {os.path.basename(audio_path)} on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return [], {}, False
                    
            except Exception as e:
                print(f"❌ Transcription attempt {attempt + 1} failed for {os.path.basename(audio_path)}: {str(e)}")
                if attempt == max_retries - 1:
                    return [], {}, False
                
                # Clear GPU cache and try again
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(1)  # Brief pause before retry
        
        return [], {}, False

    def clear_cache(self):
        """Clear model cache"""
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def switch_model_efficiently(self, new_model_size):
        """Switch to a new model efficiently"""
        if self.current_model_size == new_model_size and self.model:
            return self.model
        
        print(f"🔄 Switching model: {self.current_model_size} → {new_model_size}")
        
        # Clear current model
        self.clear_cache()
        
        # Load new model
        return self.load_model(new_model_size)

    def create_srt(self, segments, srt_path, original_duration=None):
        """Create SRT file from segments"""
        try:
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, start=1):
                    if not segment.text or not segment.text.strip():
                        continue
                        
                    f.write(f"{i}\n")
                    
                    start_time = self.seconds_to_srt_time(segment.start)
                    if i == len(segments) and original_duration is not None:
                        end_time = self.seconds_to_srt_time(original_duration)
                    else:
                        end_time = self.seconds_to_srt_time(segment.end)
                    
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment.text.strip()}\n\n")
                        
            return True
        except Exception as e:
            print(f"Error creating SRT file: {str(e)}")
            return False

    def seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

class OTIOAudioEditor(QMainWindow):
    """Main application window for OTIO Audio Editor."""    
    def __init__(self):
        """Main application window for OTIO Audio Editor."""    
        super().__init__()
        self.setup_ui()
        self.setup_connections()
        
        # Initialize variables - UPDATED for multiple files
        self.timeline = None  # Keep for backward compatibility
        self.otio_path = None  # Keep for backward compatibility
        
        # NEW: Support for multiple OTIO files
        self.timelines = {}  # Dictionary: file_path -> timeline
        self.otio_paths = []  # List of loaded OTIO file paths
        
        # NEW: Track output directories per OTIO file
        self.otio_output_dirs = {}  # Dictionary: file_path -> output_directory
        
        # CHANGED: Media file to stream mapping now includes track position
        # Key format: (media_path, local_track_index) -> selected_stream_index
        self.media_stream_mappings = {}  # Dictionary: (media_path, track_position) -> stream_index
        self.auto_mapping_enabled = True  # Flag to enable/disable auto-mapping
        
        # Rest of initialization
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(script_dir, "audio_files")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.output_dir_label = f"audio_files (in script directory)"
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audio_tracked = False
        self.clips_info = []
        self.volume_boost = 1.0
        
        self.transcription_settings = TranscriptionSettings()
        self.vocal_isolation_settings = VocalIsolationSettings()
        
        self.log("Application started. Please load OTIO file(s).")

    def setup_ui(self):
        """Set up the user interface with multiple tabs."""
        self.setWindowTitle("OTIO Audio Track & Transcription Editor")
        self.setMinimumSize(1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create main tab widget
        self.main_tabs = QTabWidget()
        main_layout.addWidget(self.main_tabs)
        
        # Tab 1: File & Audio Selection
        self.setup_file_audio_tab()
        
        # Tab 2: Transcription Settings
        self.setup_transcription_tab()
        
        # Tab 3: Audio Processing Settings
        self.setup_audio_processing_tab()
        
        # Tab 4: Progress & Output
        self.setup_progress_output_tab()
        
        # Process controls section (always visible at bottom)
        controls_layout = QHBoxLayout()
        
        self.render_button = QPushButton("Extract Audio & Transcribe")
        self.render_button.setEnabled(False)
        self.render_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.render_button.setMinimumHeight(50)
        self.render_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        controls_layout.addWidget(self.render_button)
        main_layout.addLayout(controls_layout)
        
        # Set central widget
        self.setCentralWidget(central_widget)

    def setup_file_audio_tab(self):
        """Set up the File & Audio Selection tab with enhanced controls for large datasets and auto-mapping functionality."""
        file_audio_widget = QWidget()
        layout = QVBoxLayout(file_audio_widget)
        
        # File browser section - FIXED: Removed height restrictions and improved spacing
        file_group = QGroupBox("OTIO File Management")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(10)  # Increased spacing for better readability
        
        # OTIO file selection row
        otio_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse OTIO File...")
        self.browse_button.setMinimumWidth(120)
        self.browse_button.setMinimumHeight(32)  # Increased height for better visibility
        self.browse_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Clear & Reload button
        self.clear_reload_button = QPushButton("Clear & Reload All")
        self.clear_reload_button.setMinimumWidth(120)
        self.clear_reload_button.setMinimumHeight(32)  # Increased height
        self.clear_reload_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.clear_reload_button.setEnabled(False)
        self.clear_reload_button.setToolTip("Clear all loaded OTIO files and reload them to refresh media status")
        self.clear_reload_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.otio_path_label = QLabel("No file selected")
        self.otio_path_label.setStyleSheet("""
            QLabel {
                padding: 8px; 
                background-color: #f8f9fa; 
                border: 1px solid #dee2e6; 
                border-radius: 4px; 
                font-size: 11px;
                min-height: 16px;
            }
        """)
        self.otio_path_label.setMinimumHeight(32)  # Ensure proper height
        self.otio_path_label.setWordWrap(True)  # Allow text wrapping for long paths
        
        otio_layout.addWidget(self.browse_button)
        otio_layout.addWidget(self.clear_reload_button)
        otio_layout.addWidget(self.otio_path_label, 1)
        
        # Media path search row
        search_layout = QHBoxLayout()
        self.search_media_button = QPushButton("Update Media Paths...")
        self.search_media_button.setMinimumWidth(120)
        self.search_media_button.setMinimumHeight(32)  # Increased height
        self.search_media_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.search_media_button.setEnabled(False)
        self.search_media_button.setToolTip("Browse for video files to update OTIO media paths")
        
        # Refresh Media Status button
        self.refresh_media_button = QPushButton("Refresh Status")
        self.refresh_media_button.setMinimumWidth(100)
        self.refresh_media_button.setMinimumHeight(32)  # Increased height
        self.refresh_media_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.refresh_media_button.setEnabled(False)
        self.refresh_media_button.setToolTip("Refresh media file status without reloading OTIO files")
        self.refresh_media_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.media_status_label = QLabel("Media paths not checked")
        self.media_status_label.setStyleSheet("""
            QLabel {
                padding: 8px; 
                background-color: #f8f9fa; 
                border: 1px solid #dee2e6; 
                border-radius: 4px; 
                color: gray; 
                font-size: 11px;
                min-height: 16px;
            }
        """)
        self.media_status_label.setMinimumHeight(32)  # Ensure proper height
        self.media_status_label.setWordWrap(True)  # Allow text wrapping
        
        search_layout.addWidget(self.search_media_button)
        search_layout.addWidget(self.refresh_media_button)
        search_layout.addWidget(self.media_status_label, 1)
        
        file_layout.addLayout(otio_layout)
        file_layout.addLayout(search_layout)
        file_group.setLayout(file_layout)
        
        layout.addWidget(file_group)
        
        # Audio tracks section - UPDATED for tree-based grouping
        tracks_group = QGroupBox("Audio Tracks by OTIO File")
        tracks_layout = QVBoxLayout()
        tracks_layout.setSpacing(8)
        
        # Track selection controls - enhanced for tree structure
        track_controls_layout = QHBoxLayout()
        track_controls_layout.setSpacing(8)
        
        track_label = QLabel("Track Selection:")
        track_label.setMinimumWidth(120)
        track_label.setFont(QFont("Arial", 10, QFont.Bold))
        
        self.select_all_tracks_button = QPushButton("Select All Tracks")
        self.select_all_tracks_button.setMinimumWidth(100)
        self.select_all_tracks_button.setMinimumHeight(28)
        self.select_all_tracks_button.setToolTip("Select all audio tracks from all OTIO files")
        
        self.deselect_all_tracks_button = QPushButton("Deselect All")
        self.deselect_all_tracks_button.setMinimumWidth(80)
        self.deselect_all_tracks_button.setMinimumHeight(28)
        
        # NEW: File-level selection controls
        self.select_by_otio_button = QPushButton("Select by OTIO...")
        self.select_by_otio_button.setMinimumWidth(100)
        self.select_by_otio_button.setMinimumHeight(28)
        self.select_by_otio_button.setToolTip("Select all tracks from specific OTIO files")
        self.select_by_otio_button.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a2d91;
            }
        """)
        
        # NEW: Expand/collapse controls
        self.expand_all_otio_button = QPushButton("Expand All")
        self.expand_all_otio_button.setMinimumWidth(80)
        self.expand_all_otio_button.setMinimumHeight(28)
        self.expand_all_otio_button.setToolTip("Expand all OTIO file groups")
        
        self.collapse_all_otio_button = QPushButton("Collapse All")
        self.collapse_all_otio_button.setMinimumWidth(80)
        self.collapse_all_otio_button.setMinimumHeight(28)
        self.collapse_all_otio_button.setToolTip("Collapse all OTIO file groups")
        
        track_controls_layout.addWidget(track_label)
        track_controls_layout.addWidget(self.select_all_tracks_button)
        track_controls_layout.addWidget(self.deselect_all_tracks_button)
        
        # Separator
        separator1 = QLabel("|")
        separator1.setStyleSheet("color: #ccc; font-weight: bold; font-size: 14px; margin: 0 5px;")
        track_controls_layout.addWidget(separator1)
        
        track_controls_layout.addWidget(self.select_by_otio_button)
        
        # Separator
        separator2 = QLabel("|")
        separator2.setStyleSheet("color: #ccc; font-weight: bold; font-size: 14px; margin: 0 5px;")
        track_controls_layout.addWidget(separator2)
        
        track_controls_layout.addWidget(self.expand_all_otio_button)
        track_controls_layout.addWidget(self.collapse_all_otio_button)
        
        # NEW: Auto-mapping controls
        separator3 = QLabel("|")
        separator3.setStyleSheet("color: #ccc; font-weight: bold; font-size: 14px; margin: 0 5px;")
        track_controls_layout.addWidget(separator3)
        
        self.auto_mapping_checkbox = QCheckBox("Auto-map same media")
        self.auto_mapping_checkbox.setChecked(True)
        self.auto_mapping_checkbox.setToolTip("Automatically apply stream selections to tracks using the same media file")
        track_controls_layout.addWidget(self.auto_mapping_checkbox)
        
        self.show_mappings_button = QPushButton("Show Mappings")
        self.show_mappings_button.setMinimumWidth(100)
        self.show_mappings_button.setMinimumHeight(28)
        self.show_mappings_button.setToolTip("Show current media file to stream mappings")
        self.show_mappings_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        track_controls_layout.addWidget(self.show_mappings_button)
        
        track_controls_layout.addStretch()
        
        tracks_layout.addLayout(track_controls_layout)
        
        # Audio track tree - CHANGED from QListWidget to QTreeWidget
        self.audio_track_list = QTreeWidget()
        self.audio_track_list.setHeaderLabels(["OTIO File / Track", "Clips", "Streams"])
        self.audio_track_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.audio_track_list.setMinimumHeight(200)
        self.audio_track_list.setAlternatingRowColors(True)
        self.audio_track_list.setRootIsDecorated(True)
        
        # Set column widths
        self.audio_track_list.setColumnWidth(0, 300)  # OTIO File / Track name
        self.audio_track_list.setColumnWidth(1, 80)   # Clips count
        self.audio_track_list.setColumnWidth(2, 150)  # Streams dropdown
        
        tracks_layout.addWidget(self.audio_track_list)
        
        tracks_group.setLayout(tracks_layout)
        layout.addWidget(tracks_group)
        
        # Audio cuts section with splitter (unchanged)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Audio cuts tree
        cuts_widget = QWidget()
        cuts_layout = QVBoxLayout(cuts_widget)
        cuts_layout.setContentsMargins(0, 0, 0, 0)
        
        # ENHANCED: Audio cuts controls with large dataset support
        cuts_controls_layout = QHBoxLayout()
        cuts_controls_layout.setSpacing(8)
        
        # Basic selection controls
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.setMinimumWidth(80)
        self.select_all_button.setToolTip("Select all audio cuts (works with collapsed items)")
        
        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.setMinimumWidth(80)
        
        self.invert_selection_button = QPushButton("Invert")
        self.invert_selection_button.setMinimumWidth(60)
        
        # NEW: File-level selection controls
        self.select_by_file_button = QPushButton("Select by File...")
        self.select_by_file_button.setMinimumWidth(100)
        self.select_by_file_button.setToolTip("Select all cuts from specific OTIO files")
        self.select_by_file_button.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a2d91;
            }
        """)
        
        # NEW: Range selection controls
        self.select_range_button = QPushButton("Select Range...")
        self.select_range_button.setMinimumWidth(100)
        self.select_range_button.setToolTip("Select a range of clips by index numbers")
        self.select_range_button.setStyleSheet("""
            QPushButton {
                background-color: #fd7e14;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e8590c;
            }
        """)
        
        # NEW: Expand/collapse controls for large datasets
        self.expand_all_button = QPushButton("Expand All")
        self.expand_all_button.setMinimumWidth(80)
        self.expand_all_button.setToolTip("Expand all tracks (may be slow for large datasets)")
        
        self.collapse_all_button = QPushButton("Collapse All")
        self.collapse_all_button.setMinimumWidth(80)
        self.collapse_all_button.setToolTip("Collapse all tracks for better performance")
        
        # Layout the controls in sections with better spacing
        cuts_controls_layout.addWidget(QLabel("Clip Selection:"))
        cuts_controls_layout.addWidget(self.select_all_button)
        cuts_controls_layout.addWidget(self.deselect_all_button)
        cuts_controls_layout.addWidget(self.invert_selection_button)
        
        # Separator with better styling
        separator1 = QLabel("|")
        separator1.setStyleSheet("color: #ccc; font-weight: bold; font-size: 14px; margin: 0 5px;")
        cuts_controls_layout.addWidget(separator1)
        
        cuts_controls_layout.addWidget(self.select_by_file_button)
        cuts_controls_layout.addWidget(self.select_range_button)
        
        # Separator
        separator2 = QLabel("|")
        separator2.setStyleSheet("color: #ccc; font-weight: bold; font-size: 14px; margin: 0 5px;")
        cuts_controls_layout.addWidget(separator2)
        
        cuts_controls_layout.addWidget(self.expand_all_button)
        cuts_controls_layout.addWidget(self.collapse_all_button)
        cuts_controls_layout.addStretch()
        
        cuts_layout.addLayout(cuts_controls_layout)
        
        # Audio cuts tree
        self.audio_cuts_tree = QTreeWidget()
        self.audio_cuts_tree.setHeaderLabels(["Index", "Cut/Scene", "Start Time", "Duration"])
        self.audio_cuts_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.audio_cuts_tree.setSortingEnabled(True)
        self.audio_cuts_tree.setColumnWidth(0, 60)
        self.audio_cuts_tree.setColumnWidth(1, 250)
        self.audio_cuts_tree.setColumnWidth(2, 100)
        self.audio_cuts_tree.setColumnWidth(3, 100)
        
        # Enhanced tree widget for large datasets
        self.audio_cuts_tree.setAlternatingRowColors(True)
        self.audio_cuts_tree.setRootIsDecorated(True)
        self.audio_cuts_tree.setUniformRowHeights(True)  # Better performance for large datasets
        self.audio_cuts_tree.setAnimated(False)  # Disable animations for better performance
        
        cuts_layout.addWidget(self.audio_cuts_tree)
        
        # Enhanced selection count with performance info
        selection_layout = QHBoxLayout()
        selection_layout.setSpacing(10)
        
        self.selected_count_label = QLabel("0 selected")
        self.selected_count_label.setStyleSheet("font-weight: bold; color: #2196F3; font-size: 11px;")
        
        self.clip_count_label = QLabel("0 clips total")
        self.clip_count_label.setStyleSheet("color: #666; font-size: 11px;")
        
        # NEW: Performance indicator
        self.performance_indicator = QLabel("")
        self.performance_indicator.setStyleSheet("color: #856404; font-size: 10px; font-style: italic;")
        
        # NEW: Selection tools
        self.selection_tools_label = QLabel("💡 Tips: Use 'Select by File' for quick selection of entire OTIO files")
        self.selection_tools_label.setStyleSheet("color: #17a2b8; font-size: 10px; font-style: italic;")
        self.selection_tools_label.setWordWrap(True)
        
        selection_layout.addWidget(self.selected_count_label)
        selection_layout.addWidget(self.clip_count_label)
        selection_layout.addStretch()
        selection_layout.addWidget(self.performance_indicator)
        
        # Second row for tips
        selection_layout2 = QHBoxLayout()
        selection_layout2.addWidget(self.selection_tools_label)
        selection_layout2.addStretch()
        
        cuts_layout.addLayout(selection_layout)
        cuts_layout.addLayout(selection_layout2)
        
        splitter.addWidget(cuts_widget)
        layout.addWidget(splitter)
        
        # Add to main tabs
        self.main_tabs.addTab(file_audio_widget, "📁 Files & Audio")

    def setup_transcription_tab(self):
        """Set up the Transcription Settings tab."""
        transcription_widget = QWidget()
        layout = QVBoxLayout(transcription_widget)
        
        # Enable transcription
        enable_group = QGroupBox("Transcription Control")
        enable_layout = QHBoxLayout()
        self.enable_transcription = QCheckBox("Enable Transcription")
        self.enable_transcription.setChecked(True)
        self.enable_transcription.setFont(QFont("Arial", 10, QFont.Bold))
        enable_layout.addWidget(self.enable_transcription)
        enable_layout.addStretch()
        enable_group.setLayout(enable_layout)
        layout.addWidget(enable_group)
        
        # Create scrollable area for transcription settings
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Transcription mode selection
        mode_group = QGroupBox("Transcription Mode")
        mode_layout = QVBoxLayout()
        
        self.mode_button_group = QButtonGroup()
        
        self.single_model_radio = QRadioButton("Single Model")
        self.single_model_radio.setChecked(True)
        self.single_model_radio.setToolTip("Use a single model for transcription")
        self.mode_button_group.addButton(self.single_model_radio, 0)
        
        self.comparison_radio = QRadioButton("Model Comparison")
        self.comparison_radio.setToolTip("Compare transcription results from multiple models")
        self.mode_button_group.addButton(self.comparison_radio, 1)
        
        self.voting_radio = QRadioButton("Voting System")
        self.voting_radio.setToolTip("Use all available models and vote word-by-word for best result")
        self.mode_button_group.addButton(self.voting_radio, 2)
        
        # Add descriptions
        single_desc = QLabel("• Fast processing with one selected model\n• Good for most use cases")
        single_desc.setStyleSheet("color: #666; font-size: 9px; margin-left: 20px;")
        
        comparison_desc = QLabel("• Compare multiple models side-by-side\n• Manual selection of best results")
        comparison_desc.setStyleSheet("color: #666; font-size: 9px; margin-left: 20px;")
        
        voting_desc = QLabel("• Automatic word-level voting across all models\n• Highest accuracy but slowest processing")
        voting_desc.setStyleSheet("color: #666; font-size: 9px; margin-left: 20px;")
        
        mode_layout.addWidget(self.single_model_radio)
        mode_layout.addWidget(single_desc)
        mode_layout.addWidget(self.comparison_radio)
        mode_layout.addWidget(comparison_desc)
        mode_layout.addWidget(self.voting_radio)
        mode_layout.addWidget(voting_desc)
        
        mode_group.setLayout(mode_layout)
        scroll_layout.addWidget(mode_group)
        
        # Base model selection
        base_model_group = QGroupBox("Primary Model")
        base_model_layout = QHBoxLayout()
        base_model_label = QLabel("Model:")
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItems(["large-v2", "large-v3"])
        self.base_model_combo.setCurrentText("large-v2")
        self.base_model_combo.setToolTip("Primary model used for transcription")
        base_model_layout.addWidget(base_model_label)
        base_model_layout.addWidget(self.base_model_combo)
        base_model_layout.addStretch()
        base_model_group.setLayout(base_model_layout)
        scroll_layout.addWidget(base_model_group)
        
        # Comparison models section
        self.comparison_group = QGroupBox("Additional Models for Comparison")
        self.comparison_group.setEnabled(False)
        comparison_models_layout = QVBoxLayout()
        
        self.comparison_model_checks = {}
        comparison_models = [
            ("large-v3", "Latest large model"),
            ("large-v3-turbo", "Faster large model"),
            ("turbo", "Fast general model"), 
            ("distil-large-v2", "Compressed large model"),
            ("distil-large-v3", "Latest compressed model")
        ]
        
        for model_id, model_description in comparison_models:
            checkbox = QCheckBox(f"{model_id} - {model_description}")
            checkbox.setToolTip(f"{model_id}: {model_description}")
            checkbox.setEnabled(False)
            self.comparison_model_checks[model_id] = checkbox
            comparison_models_layout.addWidget(checkbox)
        
        self.comparison_group.setLayout(comparison_models_layout)
        scroll_layout.addWidget(self.comparison_group)
        
        # Voting system settings
        self.voting_group = QGroupBox("Voting System Settings")
        self.voting_group.setEnabled(False)
        voting_layout = QVBoxLayout()
        
        voting_models_label = QLabel("Models used for voting:")
        voting_models_text = QLabel("tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, large-v3-turbo, turbo, distil-medium.en, distil-small.en, distil-large-v2, distil-large-v3")
        voting_models_text.setWordWrap(True)
        voting_models_text.setStyleSheet("color: #666; font-size: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;")
        
        voting_layout.addWidget(voting_models_label)
        voting_layout.addWidget(voting_models_text)
        
        # Voting parameters
        voting_params_layout = QVBoxLayout()
        
        # Thresholds
        thresholds_layout = QHBoxLayout()
        confidence_label = QLabel("Min Confidence:")
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setValue(0.6)
        
        agreement_label = QLabel("Min Agreement:")
        self.agreement_spin = QDoubleSpinBox()
        self.agreement_spin.setRange(0.0, 1.0)
        self.agreement_spin.setSingleStep(0.1)
        self.agreement_spin.setValue(0.5)
        
        thresholds_layout.addWidget(confidence_label)
        thresholds_layout.addWidget(self.confidence_spin)
        thresholds_layout.addWidget(agreement_label)
        thresholds_layout.addWidget(self.agreement_spin)
        thresholds_layout.addStretch()
        
        # English preference
        english_layout = QHBoxLayout()
        self.prefer_english_check = QCheckBox("Prefer English Transcriptions")
        self.prefer_english_check.setChecked(True)
        
        english_bias_label = QLabel("English Bias:")
        self.english_bias_spin = QDoubleSpinBox()
        self.english_bias_spin.setRange(1.0, 2.0)
        self.english_bias_spin.setSingleStep(0.1)
        self.english_bias_spin.setValue(1.2)
        
        english_layout.addWidget(self.prefer_english_check)
        english_layout.addWidget(english_bias_label)
        english_layout.addWidget(self.english_bias_spin)
        english_layout.addStretch()
        
        voting_params_layout.addLayout(thresholds_layout)
        voting_params_layout.addLayout(english_layout)
        voting_layout.addLayout(voting_params_layout)
        
        self.voting_group.setLayout(voting_layout)
        scroll_layout.addWidget(self.voting_group)
        
        # Basic transcription settings
        basic_group = QGroupBox("Basic Settings")
        basic_layout = QVBoxLayout()
        
        # Language and device
        lang_device_layout = QHBoxLayout()
        
        language_label = QLabel("Language:")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["auto", "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"])
        self.language_combo.setCurrentText("en")
        
        device_label = QLabel("Device:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        if torch.cuda.is_available():
            self.device_combo.setCurrentText("cuda")
        
        lang_device_layout.addWidget(language_label)
        lang_device_layout.addWidget(self.language_combo)
        lang_device_layout.addWidget(device_label)
        lang_device_layout.addWidget(self.device_combo)
        lang_device_layout.addStretch()
        
        # SRT output options
        srt_layout = QHBoxLayout()
        self.create_srt_check = QCheckBox("Create Individual SRTs")
        self.create_srt_check.setChecked(True)
        self.create_combined_srt_check = QCheckBox("Create Combined SRT")
        self.create_combined_srt_check.setChecked(True)
        srt_layout.addWidget(self.create_srt_check)
        srt_layout.addWidget(self.create_combined_srt_check)
        srt_layout.addStretch()
        
        basic_layout.addLayout(lang_device_layout)
        basic_layout.addLayout(srt_layout)
        basic_group.setLayout(basic_layout)
        scroll_layout.addWidget(basic_group)
        
        # Advanced settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QVBoxLayout()
        
        # Alignment settings
        alignment_layout = QHBoxLayout()
        self.improved_alignment_check = QCheckBox("Use Improved Alignment")
        self.improved_alignment_check.setChecked(True)
        self.improved_alignment_check.setToolTip("Uses WAV2VEC2 model for better timestamp accuracy")
        
        batch_label = QLabel("Batch Size:")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMinimum(1)
        self.batch_size_spin.setMaximum(32)
        self.batch_size_spin.setValue(4)
        
        alignment_layout.addWidget(self.improved_alignment_check)
        alignment_layout.addWidget(batch_label)
        alignment_layout.addWidget(self.batch_size_spin)
        alignment_layout.addStretch()
        
        # Compute type
        compute_layout = QHBoxLayout()
        compute_label = QLabel("Compute Type:")
        self.compute_type_combo = QComboBox()
        self.compute_type_combo.addItems(["float16", "float32", "int8"])
        self.compute_type_combo.setCurrentText("int8")
        self.compute_type_combo.setToolTip("Use int8 for CPU or low memory GPUs")
        
        compute_layout.addWidget(compute_label)
        compute_layout.addWidget(self.compute_type_combo)
        compute_layout.addStretch()
        
        advanced_layout.addLayout(alignment_layout)
        advanced_layout.addLayout(compute_layout)
        advanced_group.setLayout(advanced_layout)
        scroll_layout.addWidget(advanced_group)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Add to main tabs
        self.main_tabs.addTab(transcription_widget, "🎤 Transcription Settings")

    def setup_audio_processing_tab(self):
        """Set up the Audio Processing Settings tab."""
        audio_widget = QWidget()
        layout = QVBoxLayout(audio_widget)
        
        # Audio normalization settings
        audio_settings_group = QGroupBox("Audio Normalization")
        audio_settings_layout = QVBoxLayout()
        
        self.audio_normalize_check = QCheckBox("Enable EBU R128 Loudness Normalization")
        self.audio_normalize_check.setChecked(True)
        self.audio_normalize_check.setToolTip("Normalizes audio to maximize volume without distortion")
        
        normalize_desc = QLabel("• Standardizes audio levels across all clips\n• Improves transcription accuracy for quiet audio\n• Uses broadcast standard EBU R128")
        normalize_desc.setStyleSheet("color: #666; font-size: 10px; margin-left: 20px; padding: 10px;")
        
        audio_settings_layout.addWidget(self.audio_normalize_check)
        audio_settings_layout.addWidget(normalize_desc)
        audio_settings_group.setLayout(audio_settings_layout)
        layout.addWidget(audio_settings_group)
        
        # Vocal Isolation settings
        vocal_isolation_group = QGroupBox("Vocal Isolation (Experimental)")
        vocal_isolation_layout = QVBoxLayout()
        
        # Enable vocal isolation
        self.enable_isolation = QCheckBox("Enable Vocal Isolation")
        self.enable_isolation.setChecked(False)
        vocal_isolation_layout.addWidget(self.enable_isolation)
        
        isolation_desc = QLabel("• Uses AI to separate vocals from background music\n• Improves transcription accuracy for music videos\n• Requires additional processing time")
        isolation_desc.setStyleSheet("color: #666; font-size: 10px; margin-left: 20px; padding: 10px;")
        vocal_isolation_layout.addWidget(isolation_desc)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Demucs Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra", "mdx_extra_q"])
        self.model_combo.setCurrentText("htdemucs")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        vocal_isolation_layout.addLayout(model_layout)
        
        # Output options
        output_layout = QHBoxLayout()
        self.isolate_vocals_check = QCheckBox("Extract Vocals")
        self.isolate_vocals_check.setChecked(True)
        self.isolate_instruments_check = QCheckBox("Extract Instruments")
        self.isolate_instruments_check.setChecked(False)
        output_layout.addWidget(self.isolate_vocals_check)
        output_layout.addWidget(self.isolate_instruments_check)
        output_layout.addStretch()
        vocal_isolation_layout.addLayout(output_layout)
        
        vocal_isolation_group.setLayout(vocal_isolation_layout)
        layout.addWidget(vocal_isolation_group)
        
        # Output Directory settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        
        dir_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory:")
        self.output_path_label = QLabel("audio_files (in script directory)")
        self.output_path_label.setStyleSheet("padding: 8px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;")
        self.output_browse_button = QPushButton("Browse...")
        self.output_browse_button.setMinimumWidth(100)
        
        dir_layout.addWidget(self.output_label)
        dir_layout.addWidget(self.output_path_label, 1)
        dir_layout.addWidget(self.output_browse_button)
        
        output_layout.addLayout(dir_layout)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        layout.addStretch()
        
        # Add to main tabs
        self.main_tabs.addTab(audio_widget, "🔧 Audio Processing")

    def setup_progress_output_tab(self):
        """Set up the Progress & Output tab."""
        progress_widget = QWidget()
        layout = QVBoxLayout(progress_widget)
        
        # Progress section
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.progress_bar)
        
        # Status labels
        status_layout = QHBoxLayout()
        self.current_operation_label = QLabel("Ready")
        self.current_operation_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        
        self.processing_stats_label = QLabel("")
        self.processing_stats_label.setStyleSheet("color: #666; font-size: 10px;")
        
        status_layout.addWidget(self.current_operation_label)
        status_layout.addStretch()
        status_layout.addWidget(self.processing_stats_label)
        
        progress_layout.addLayout(status_layout)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Log section
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        
        # Log controls
        log_controls_layout = QHBoxLayout()
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.setMaximumWidth(100)
        self.save_log_button = QPushButton("Save Log...")
        self.save_log_button.setMaximumWidth(100)
        
        self.log_filter_combo = QComboBox()
        self.log_filter_combo.addItems(["All Messages", "Info Only", "Errors Only"])
        self.log_filter_combo.setMaximumWidth(150)
        
        log_controls_layout.addWidget(QLabel("Filter:"))
        log_controls_layout.addWidget(self.log_filter_combo)
        log_controls_layout.addStretch()
        log_controls_layout.addWidget(self.clear_log_button)
        log_controls_layout.addWidget(self.save_log_button)
        
        log_layout.addLayout(log_controls_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(300)
        self.log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Output files section
        output_files_group = QGroupBox("Generated Files")
        output_files_layout = QVBoxLayout()
        
        self.output_files_list = QListWidget()
        self.output_files_list.setMaximumHeight(150)
        self.output_files_list.setStyleSheet("""
            QListWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
        """)
        
        output_files_controls_layout = QHBoxLayout()
        self.open_output_folder_button = QPushButton("Open Output Folder")
        self.refresh_files_button = QPushButton("Refresh")
        
        output_files_controls_layout.addWidget(self.open_output_folder_button)
        output_files_controls_layout.addWidget(self.refresh_files_button)
        output_files_controls_layout.addStretch()
        
        output_files_layout.addWidget(self.output_files_list)
        output_files_layout.addLayout(output_files_controls_layout)
        output_files_group.setLayout(output_files_layout)
        layout.addWidget(output_files_group)
        
        # Add to main tabs
        self.main_tabs.addTab(progress_widget, "📊 Progress & Output")

    def setup_connections(self):
        """Set up signal/slot connections for the new tab structure with enhanced auto-mapping functionality."""
        # File & Audio tab connections
        self.browse_button.clicked.connect(self.browse_otio)
        self.search_media_button.clicked.connect(self.search_and_update_media_paths)
        
        # UPDATED: Tree widget selection changed signal instead of list widget
        self.audio_track_list.itemSelectionChanged.connect(self.display_selected_audio_cuts)
        
        # Enhanced OTIO file management connections
        self.clear_reload_button.clicked.connect(self.clear_and_reload_otio_files)
        self.refresh_media_button.clicked.connect(self.refresh_media_status)
        
        # Audio cuts selection connections (ENHANCED)
        self.select_all_button.clicked.connect(self.select_all_cuts)
        self.deselect_all_button.clicked.connect(self.deselect_all_cuts)
        self.invert_selection_button.clicked.connect(self.invert_cut_selection)
        
        # NEW: Enhanced selection controls
        self.select_by_file_button.clicked.connect(self.show_file_selection_dialog)
        self.select_range_button.clicked.connect(self.select_range_dialog)
        self.expand_all_button.clicked.connect(self.expand_all_tracks)
        self.collapse_all_button.clicked.connect(self.collapse_all_tracks)
        
        # UPDATED: Tree-based track selection connections
        self.select_all_tracks_button.clicked.connect(self.select_all_tracks)
        self.deselect_all_tracks_button.clicked.connect(self.deselect_all_tracks)
        
        # NEW: OTIO-level track selection
        self.select_by_otio_button.clicked.connect(self.show_otio_track_selection_dialog)
        self.expand_all_otio_button.clicked.connect(self.expand_all_otio_tracks)
        self.collapse_all_otio_button.clicked.connect(self.collapse_all_otio_tracks)
        
        # NEW: Auto-mapping controls
        self.auto_mapping_checkbox.stateChanged.connect(self.toggle_auto_mapping)
        self.show_mappings_button.clicked.connect(self.show_media_mappings_dialog)
        
        # Selection change tracking (ENHANCED)
        self.audio_cuts_tree.itemSelectionChanged.connect(self.update_selection_count)
        
        # Main process button
        self.render_button.clicked.connect(self.start_audio_rendering)
        
        # Transcription settings connections
        self.enable_transcription.stateChanged.connect(self.update_transcription_settings)
        self.mode_button_group.buttonClicked.connect(self.on_transcription_mode_changed)
        self.base_model_combo.currentTextChanged.connect(self.update_transcription_settings)
        self.language_combo.currentTextChanged.connect(self.update_transcription_settings)
        self.device_combo.currentTextChanged.connect(self.update_transcription_settings)
        self.create_srt_check.stateChanged.connect(self.update_transcription_settings)
        self.create_combined_srt_check.stateChanged.connect(self.update_transcription_settings)
        self.improved_alignment_check.stateChanged.connect(self.update_transcription_settings)
        self.batch_size_spin.valueChanged.connect(self.update_transcription_settings)
        self.compute_type_combo.currentTextChanged.connect(self.update_transcription_settings)
        
        # Model comparison connections
        for model_id, checkbox in self.comparison_model_checks.items():
            checkbox.stateChanged.connect(self.update_transcription_settings)
        
        # Voting system connections
        self.confidence_spin.valueChanged.connect(self.update_transcription_settings)
        self.agreement_spin.valueChanged.connect(self.update_transcription_settings)
        self.prefer_english_check.stateChanged.connect(self.update_transcription_settings)
        self.english_bias_spin.valueChanged.connect(self.update_transcription_settings)
        
        # Audio processing connections (vocal isolation settings)
        self.enable_isolation.stateChanged.connect(self.update_vocal_isolation_settings)
        self.model_combo.currentTextChanged.connect(self.update_vocal_isolation_settings)
        self.isolate_vocals_check.stateChanged.connect(self.update_vocal_isolation_settings)
        self.isolate_instruments_check.stateChanged.connect(self.update_vocal_isolation_settings)
        
        # Audio normalization connection
        if hasattr(self, 'audio_normalize_check'):
            self.audio_normalize_check.stateChanged.connect(self.update_transcription_settings)
        
        # Output settings connections
        self.output_browse_button.clicked.connect(self.browse_audio_output)
        
        # Progress & Output tab connections
        self.clear_log_button.clicked.connect(self.clear_log)
        self.save_log_button.clicked.connect(self.save_log)
        self.log_filter_combo.currentTextChanged.connect(self.apply_log_filter)
        self.open_output_folder_button.clicked.connect(self.open_output_folder)
        self.refresh_files_button.clicked.connect(self.refresh_output_files)
        
        # Tab change connection to auto-switch to progress tab when processing starts
        self.main_tabs.currentChanged.connect(self.on_tab_changed)

    def toggle_auto_mapping(self, state):
        """Toggle auto-mapping functionality"""
        from PyQt5.QtCore import Qt
        
        self.auto_mapping_enabled = state == Qt.Checked
        status = "enabled" if self.auto_mapping_enabled else "disabled"
        self.log(f"Auto-mapping {status}")
        
        if self.auto_mapping_enabled:
            self.log("When you select an audio stream, it will automatically be applied to all tracks using the same media file")
        else:
            self.log("Auto-mapping disabled - stream selections will only affect the current track")

    def show_media_mappings_dialog(self):
        """Show dialog with current media file mappings"""
        dialog = MediaMappingsDialog(self.media_stream_mappings, self.timelines, self)
        dialog.exec_()

    def show_otio_track_selection_dialog(self):
        """Show dialog to select all tracks from specific OTIO files."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QPushButton, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Tracks by OTIO File")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Add instructions
        instructions = QLabel("Select which OTIO files to include all tracks from:")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("margin-bottom: 10px; font-weight: bold;")
        layout.addWidget(instructions)
        
        # Get OTIO files and their track counts
        otio_files = []
        for i in range(self.audio_track_list.topLevelItemCount()):
            otio_item = self.audio_track_list.topLevelItem(i)
            otio_data = otio_item.data(0, Qt.UserRole)
            
            if otio_data and otio_data.get("type") == "otio_file":
                track_count = otio_item.childCount()
                file_name = otio_data.get("file_name", "Unknown")
                otio_files.append((file_name, i, track_count))
        
        if not otio_files:
            QMessageBox.information(self, "No Files", "No OTIO files found.")
            return
        
        # Create checkboxes for each OTIO file
        checkboxes = []
        for file_name, file_index, track_count in otio_files:
            checkbox = QCheckBox(f"{file_name} ({track_count} tracks)")
            checkboxes.append((checkbox, file_index))
            layout.addWidget(checkbox)
        
        # Quick selection buttons
        quick_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb, _ in checkboxes])
        
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(lambda: [cb.setChecked(False) for cb, _ in checkboxes])
        
        quick_layout.addWidget(select_all_btn)
        quick_layout.addWidget(select_none_btn)
        quick_layout.addStretch()
        layout.addLayout(quick_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QDialog.Accepted:
            # Select tracks from checked OTIO files
            selected_files = [file_index for checkbox, file_index in checkboxes if checkbox.isChecked()]
            if selected_files:
                self.select_tracks_by_otio_indices(selected_files)
            else:
                QMessageBox.information(self, "No Selection", "No OTIO files were selected.")

    def select_tracks_by_otio_indices(self, otio_indices):
        """Select all tracks from specified OTIO file indices."""
        selected_count = 0
        
        for otio_index in otio_indices:
            if otio_index < self.audio_track_list.topLevelItemCount():
                otio_item = self.audio_track_list.topLevelItem(otio_index)
                
                # Select all tracks under this OTIO file
                for i in range(otio_item.childCount()):
                    track_item = otio_item.child(i)
                    track_data = track_item.data(0, Qt.UserRole)
                    
                    if isinstance(track_data, int):  # This is a track item
                        track_item.setSelected(True)
                        selected_count += 1
        
        self.log(f"Selected {selected_count} tracks from {len(otio_indices)} OTIO files")

    def expand_all_otio_tracks(self):
        """Expand all OTIO file groups to show their tracks."""
        for i in range(self.audio_track_list.topLevelItemCount()):
            otio_item = self.audio_track_list.topLevelItem(i)
            otio_item.setExpanded(True)
        
        self.log("Expanded all OTIO file groups")

    def collapse_all_otio_tracks(self):
        """Collapse all OTIO file groups to hide their tracks."""
        for i in range(self.audio_track_list.topLevelItemCount()):
            otio_item = self.audio_track_list.topLevelItem(i)
            otio_item.setExpanded(False)
        
        self.log("Collapsed all OTIO file groups")

    def select_all_tracks(self):
        """Select all audio tracks from all OTIO files - UPDATED for tree structure."""
        selected_count = 0
        
        def select_tracks_in_item(item):
            nonlocal selected_count
            
            item_data = item.data(0, Qt.UserRole)
            if isinstance(item_data, int):  # This is a track item
                item.setSelected(True)
                selected_count += 1
            elif isinstance(item_data, dict) and item_data.get("type") == "otio_file":
                # This is an OTIO file item, process its children
                for i in range(item.childCount()):
                    select_tracks_in_item(item.child(i))
            
            # Process children for any other type
            for i in range(item.childCount()):
                child = item.child(i)
                child_data = child.data(0, Qt.UserRole)
                if isinstance(child_data, int):  # Track item
                    select_tracks_in_item(child)
        
        # Process all top-level items
        for i in range(self.audio_track_list.topLevelItemCount()):
            select_tracks_in_item(self.audio_track_list.topLevelItem(i))
        
        self.log(f"Selected {selected_count} audio tracks")

    def deselect_all_tracks(self):
        """Deselect all audio tracks - UPDATED for tree structure."""
        def deselect_tracks_in_item(item):
            item.setSelected(False)
            
            # Process children
            for i in range(item.childCount()):
                deselect_tracks_in_item(item.child(i))
        
        # Process all top-level items
        for i in range(self.audio_track_list.topLevelItemCount()):
            deselect_tracks_in_item(self.audio_track_list.topLevelItem(i))
        
        self.log("Deselected all audio tracks")

    def collapse_all_tracks(self):
        """Collapse all tracks for better performance."""
        for i in range(self.audio_cuts_tree.topLevelItemCount()):
            file_item = self.audio_cuts_tree.topLevelItem(i)
            file_item.setExpanded(True)  # Keep file headers expanded
            
            # Collapse all tracks within each file
            for j in range(file_item.childCount()):
                track_item = file_item.child(j)
                track_item.setExpanded(False)
        
        self.log("Collapsed all tracks (file headers remain expanded)")

    def expand_all_tracks(self):
        """Expand all tracks - with warning for large datasets."""
        total_clips = self.count_total_clips()
        
        if total_clips > 500:
            reply = QMessageBox.question(
                self, 
                "Expand All Tracks", 
                f"You have {total_clips} total clips. Expanding all tracks may cause performance issues.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        self.audio_cuts_tree.expandAll()
        self.log(f"Expanded all tracks ({total_clips} clips)")

    def count_total_clips(self):
        """Count total number of clips in the tree."""
        total = 0
        
        def count_clips_in_item(item):
            nonlocal total
            
            item_data = item.data(0, Qt.UserRole)
            if item_data and item_data.get("type") == "cut":
                total += 1
            
            for i in range(item.childCount()):
                count_clips_in_item(item.child(i))
        
        for i in range(self.audio_cuts_tree.topLevelItemCount()):
            count_clips_in_item(self.audio_cuts_tree.topLevelItem(i))
        
        return total

    def show_file_selection_dialog(self):
        """Show dialog to select cuts by OTIO file."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QPushButton, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Cuts by OTIO File")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Add instructions
        instructions = QLabel("Select which OTIO files to include all clips from:")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("margin-bottom: 10px; font-weight: bold;")
        layout.addWidget(instructions)
        
        # Get unique OTIO files
        otio_files = []
        for i in range(self.audio_cuts_tree.topLevelItemCount()):
            file_item = self.audio_cuts_tree.topLevelItem(i)
            file_data = file_item.data(0, Qt.UserRole)
            if file_data and file_data.get("type") == "file":
                # Count clips in this file
                clip_count = 0
                def count_clips_in_item(item):
                    nonlocal clip_count
                    for j in range(item.childCount()):
                        child = item.child(j)
                        child_data = child.data(0, Qt.UserRole)
                        if child_data and child_data.get("type") == "cut":
                            clip_count += 1
                        else:
                            count_clips_in_item(child)
                
                count_clips_in_item(file_item)
                otio_files.append((file_data["file_name"], file_data["file_index"], clip_count))
        
        if not otio_files:
            QMessageBox.information(self, "No Files", "No OTIO files found.")
            return
        
        # Create checkboxes for each file
        checkboxes = []
        for file_name, file_index, clip_count in otio_files:
            checkbox = QCheckBox(f"{file_name} ({clip_count} clips)")
            checkboxes.append((checkbox, file_index))
            layout.addWidget(checkbox)
        
        # Quick selection buttons
        quick_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb, _ in checkboxes])
        
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(lambda: [cb.setChecked(False) for cb, _ in checkboxes])
        
        quick_layout.addWidget(select_all_btn)
        quick_layout.addWidget(select_none_btn)
        quick_layout.addStretch()
        layout.addLayout(quick_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QDialog.Accepted:
            # Select cuts from checked files
            selected_files = [file_index for checkbox, file_index in checkboxes if checkbox.isChecked()]
            if selected_files:
                self.select_cuts_by_file_indices(selected_files)
            else:
                QMessageBox.information(self, "No Selection", "No files were selected.")

    def select_cuts_by_file_indices(self, file_indices):
        """Select all cuts from specified file indices."""
        self.audio_cuts_tree.setUpdatesEnabled(False)
        
        try:
            selected_count = 0
            
            def select_cuts_in_file(item, target_file_indices):
                nonlocal selected_count
                
                item_data = item.data(0, Qt.UserRole)
                
                if item_data:
                    if item_data.get("type") == "cut":
                        file_index = item_data.get("file_index")
                        if file_index in target_file_indices:
                            item.setSelected(True)
                            selected_count += 1
                        else:
                            item.setSelected(False)
                    elif item_data.get("type") == "file":
                        file_index = item_data.get("file_index")
                        if file_index in target_file_indices:
                            # Select all cuts in this file
                            for i in range(item.childCount()):
                                select_cuts_in_file(item.child(i), target_file_indices)
                            return  # Don't process children again
                
                # Process children
                for i in range(item.childCount()):
                    select_cuts_in_file(item.child(i), target_file_indices)
            
            # Process all items
            for i in range(self.audio_cuts_tree.topLevelItemCount()):
                select_cuts_in_file(self.audio_cuts_tree.topLevelItem(i), file_indices)
            
            self.log(f"Selected {selected_count} cuts from {len(file_indices)} files")
            
        finally:
            self.audio_cuts_tree.setUpdatesEnabled(True)
            self.update_selection_count()

    def select_range_dialog(self):
        """Show dialog to select a range of clips by index."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QSpinBox, QLabel, QPushButton, QDialogButtonBox
        
        total_clips = self.count_total_clips()
        if total_clips == 0:
            QMessageBox.information(self, "No Clips", "No clips are currently loaded.")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Range of Clips")
        dialog.setMinimumWidth(350)
        
        layout = QVBoxLayout(dialog)
        
        # Info label
        info_label = QLabel(f"Select a range from the {total_clips} total clips:")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Range inputs
        range_layout = QHBoxLayout()
        
        range_layout.addWidget(QLabel("From clip #:"))
        start_spin = QSpinBox()
        start_spin.setMinimum(1)
        start_spin.setMaximum(total_clips)
        start_spin.setValue(1)
        range_layout.addWidget(start_spin)
        
        range_layout.addWidget(QLabel("To clip #:"))
        end_spin = QSpinBox()
        end_spin.setMinimum(1)
        end_spin.setMaximum(total_clips)
        end_spin.setValue(min(100, total_clips))  # Default to first 100 or total
        range_layout.addWidget(end_spin)
        
        layout.addLayout(range_layout)
        
        # Quick selection buttons
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel("Quick select:"))
        
        first_100_btn = QPushButton("First 100")
        first_100_btn.clicked.connect(lambda: self.set_range_values(start_spin, end_spin, 1, min(100, total_clips)))
        quick_layout.addWidget(first_100_btn)
        
        last_100_btn = QPushButton("Last 100")
        last_100_btn.clicked.connect(lambda: self.set_range_values(start_spin, end_spin, max(1, total_clips-99), total_clips))
        quick_layout.addWidget(last_100_btn)
        
        middle_btn = QPushButton("Middle 100")
        middle_start = max(1, (total_clips // 2) - 50)
        middle_end = min(total_clips, middle_start + 99)
        middle_btn.clicked.connect(lambda: self.set_range_values(start_spin, end_spin, middle_start, middle_end))
        quick_layout.addWidget(middle_btn)
        
        quick_layout.addStretch()
        layout.addLayout(quick_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QDialog.Accepted:
            start_index = start_spin.value()
            end_index = end_spin.value()
            
            if start_index > end_index:
                start_index, end_index = end_index, start_index
            
            self.select_clips_by_range(start_index, end_index)

    def set_range_values(self, start_spin, end_spin, start, end):
        """Helper to set range values in the dialog."""
        start_spin.setValue(start)
        end_spin.setValue(end)

    def select_clips_by_range(self, start_index, end_index):
        """Select clips within the specified index range."""
        self.audio_cuts_tree.setUpdatesEnabled(False)
        
        try:
            selected_count = 0
            
            def select_clips_in_range(item):
                nonlocal selected_count
                
                item_data = item.data(0, Qt.UserRole)
                if item_data and item_data.get("type") == "cut":
                    global_index = item_data.get("global_index", 0)
                    
                    if start_index <= global_index <= end_index:
                        item.setSelected(True)
                        selected_count += 1
                    else:
                        item.setSelected(False)
                
                # Process children
                for i in range(item.childCount()):
                    select_clips_in_range(item.child(i))
            
            # Process all items
            for i in range(self.audio_cuts_tree.topLevelItemCount()):
                select_clips_in_range(self.audio_cuts_tree.topLevelItem(i))
            
            self.log(f"Selected {selected_count} clips in range {start_index}-{end_index}")
            
        finally:
            self.audio_cuts_tree.setUpdatesEnabled(True)
            self.update_selection_count()

    def update_performance_indicator(self, total_clips):
        """Update the performance indicator based on dataset size."""
        if not hasattr(self, 'performance_indicator'):
            return
            
        if total_clips > 1000:
            self.performance_indicator.setText("⚠ Large dataset - use collapsed view for better performance")
            self.performance_indicator.setStyleSheet("color: #dc3545; font-size: 10px; font-style: italic;")
        elif total_clips > 500:
            self.performance_indicator.setText("⚡ Medium dataset - consider using selection tools")
            self.performance_indicator.setStyleSheet("color: #fd7e14; font-size: 10px; font-style: italic;")
        elif total_clips > 100:
            self.performance_indicator.setText("✓ Manageable dataset size")
            self.performance_indicator.setStyleSheet("color: #28a745; font-size: 10px; font-style: italic;")
        else:
            self.performance_indicator.setText("")

    def refresh_media_status(self):
        """Refresh media file status without reloading OTIO files."""
        if not self.timelines:
            return
        
        self.log("Refreshing media file status...")
        self.media_status_label.setText("Checking...")
        
        # Force a fresh check of media files
        QTimer.singleShot(100, self.check_media_files_status)  # Small delay for UI update

    def clear_and_reload_otio_files(self):
        """Clear all loaded OTIO files and reload them to refresh everything."""
        if not self.otio_paths:
            return
        
        # Store the current file paths
        current_paths = self.otio_paths.copy()
        
        # Show progress for user feedback
        reply = QMessageBox.question(
            self, 
            "Clear & Reload OTIO Files", 
            f"This will clear and reload all {len(current_paths)} OTIO files to refresh the media status.\n\n"
            "Any unsaved changes will be lost. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self.log("Clearing and reloading all OTIO files...")
        
        # Clear everything
        self.timelines.clear()
        self.otio_paths.clear()
        self.audio_track_list.clear()
        self.audio_cuts_tree.clear()
        
        # Update UI to show cleared state
        self.otio_path_label.setText("Reloading...")
        self.media_status_label.setText("Reloading...")
        
        # Disable buttons temporarily
        self.enable_ui_elements(False)
        
        # Reload all files
        successful_loads = 0
        failed_loads = []
        
        for file_path in current_paths:
            self.log(f"Reloading: {os.path.basename(file_path)}")
            if self.load_single_otio_file(file_path):
                successful_loads += 1
            else:
                failed_loads.append(os.path.basename(file_path))
        
        if successful_loads > 0:
            # Update UI with reloaded data
            self.update_ui_after_loading_multiple_files()
            self.log(f"✓ Successfully reloaded {successful_loads} of {len(current_paths)} OTIO files.")
            
            if failed_loads:
                self.log(f"⚠ Failed to reload: {', '.join(failed_loads)}", error=True)
                QMessageBox.warning(
                    self, 
                    "Partial Reload", 
                    f"Successfully reloaded {successful_loads} files.\n\n"
                    f"Failed to reload: {', '.join(failed_loads)}"
                )
            else:
                QMessageBox.information(
                    self, 
                    "Reload Complete", 
                    f"Successfully reloaded all {successful_loads} OTIO files.\n\n"
                    "Media status has been refreshed."
                )
        else:
            self.log("❌ Failed to reload any OTIO files.", error=True)
            QMessageBox.critical(
                self, 
                "Reload Failed", 
                "Failed to reload any OTIO files. Please browse and load them again."
            )

    def on_tab_changed(self, index):
        """Handle tab changes."""
        # Auto-switch to progress tab when processing starts
        tab_name = self.main_tabs.tabText(index)
        
        # Refresh output files when switching to progress tab
        if "Progress" in tab_name:
            self.refresh_output_files()

    def update_selection_count(self):
        """Enhanced update selection count with performance indicators and file breakdown."""
        selected_count = 0
        total_count = 0
        selected_by_file = {}
        
        # Count selections by file with enhanced organization
        for file_index in range(self.audio_cuts_tree.topLevelItemCount()):
            file_item = self.audio_cuts_tree.topLevelItem(file_index)
            file_data = file_item.data(0, Qt.UserRole)
            
            # Skip separator items
            if not file_data:
                continue
                
            if file_data and file_data.get("type") == "file":
                file_name = file_data.get("file_name", "Unknown")
                file_index_num = file_data.get("file_index", 0)
                display_name = f"File {file_index_num + 1}: {file_name}"
                selected_by_file[display_name] = {"selected": 0, "total": 0}
                
                def count_items_in_tree(item, file_stats):
                    nonlocal selected_count, total_count
                    
                    for i in range(item.childCount()):
                        child = item.child(i)
                        child_data = child.data(0, Qt.UserRole)
                        
                        if child_data and child_data.get("type") == "cut":
                            total_count += 1
                            file_stats["total"] += 1
                            if child.isSelected():
                                selected_count += 1
                                file_stats["selected"] += 1
                        else:
                            # Recursively count in sub-items
                            count_items_in_tree(child, file_stats)
                
                count_items_in_tree(file_item, selected_by_file[display_name])
        
        # Update main counter with file context
        if len(selected_by_file) > 1:
            self.selected_count_label.setText(f"{selected_count} of {total_count} selected across {len(selected_by_file)} files")
            self.clip_count_label.setText(f"{total_count} clips from {len(self.otio_paths)} OTIO files")
        else:
            self.selected_count_label.setText(f"{selected_count} of {total_count} selected")
            self.clip_count_label.setText(f"{total_count} clips from {len(self.otio_paths)} files")
        
        # Update performance indicator
        self.update_performance_indicator(total_count)
        
        # Enhanced render button text with file awareness
        if selected_count > 0:
            files_with_selections = sum(1 for file_stats in selected_by_file.values() if file_stats["selected"] > 0)
            
            if self.transcription_settings.enabled:
                mode = self.mode_button_group.checkedId()
                if mode == 1:  # Comparison
                    comparison_count = len([cb for cb in self.comparison_model_checks.values() if cb.isChecked()])
                    self.render_button.setText(f"Process {selected_count} Clips from {files_with_selections} Files & Compare {comparison_count + 1} Models")
                elif mode == 2:  # Voting
                    model_count = len(self.transcription_settings.voting_models)
                    self.render_button.setText(f"Process {selected_count} Clips from {files_with_selections} Files & Vote with {model_count} Models")
                else:  # Single
                    self.render_button.setText(f"Process {selected_count} Clips from {files_with_selections} Files & Transcribe")
            else:
                self.render_button.setText(f"Extract {selected_count} Clips from {files_with_selections} Files")
            
            # Update selection tools tip based on dataset size
            if total_count > 500:
                if selected_count < total_count * 0.1:  # Less than 10% selected
                    self.selection_tools_label.setText("💡 Tip: Use 'Select by File' or 'Select Range' for faster selection with large datasets")
                else:
                    self.selection_tools_label.setText(f"✓ {selected_count}/{total_count} clips selected ({selected_count/total_count*100:.1f}%)")
            elif total_count > 100:
                self.selection_tools_label.setText("💡 Tips: Use 'Select by File' for quick selection of entire OTIO files")
            else:
                self.selection_tools_label.setText("")
                
        else:
            self.render_button.setText("Select clips to process")
            if total_count > 100:
                self.selection_tools_label.setText("💡 Tips: Use 'Select All', 'Select by File', or 'Select Range' to quickly select clips")
            else:
                self.selection_tools_label.setText("")
        
        # Show file breakdown in tooltip if multiple files
        if len(selected_by_file) > 1:
            tooltip_parts = []
            for file_name, stats in selected_by_file.items():
                if stats["selected"] > 0:
                    tooltip_parts.append(f"{file_name}: {stats['selected']}/{stats['total']} selected")
            
            if tooltip_parts:
                self.render_button.setToolTip("Selection breakdown:\n" + "\n".join(tooltip_parts))
            else:
                self.render_button.setToolTip("")
        else:
            self.render_button.setToolTip("")

    def on_transcription_mode_changed(self):
        """Handle transcription mode changes"""
        selected_mode = self.mode_button_group.checkedId()
        
        # Enable/disable appropriate sections
        if selected_mode == 0:  # Single Model
            self.comparison_group.setEnabled(False)
            self.voting_group.setEnabled(False)
            for checkbox in self.comparison_model_checks.values():
                checkbox.setEnabled(False)
        elif selected_mode == 1:  # Model Comparison
            self.comparison_group.setEnabled(True)
            self.voting_group.setEnabled(False)
            for checkbox in self.comparison_model_checks.values():
                checkbox.setEnabled(True)
        elif selected_mode == 2:  # Voting System
            self.comparison_group.setEnabled(False)
            self.voting_group.setEnabled(True)
            for checkbox in self.comparison_model_checks.values():
                checkbox.setEnabled(False)
        
        # Update settings
        self.update_transcription_settings()

    def update_transcription_settings(self):
        """Update transcription settings from UI controls"""
        self.transcription_settings.enabled = self.enable_transcription.isChecked()
        
        # Determine transcription mode
        selected_mode = self.mode_button_group.checkedId()
        
        if selected_mode == 0:  # Single Model
            self.transcription_settings.enable_model_comparison = False
            self.transcription_settings.enable_voting_system = False
            self.transcription_settings.comparison_models = []
            self.transcription_settings.model_size = self.base_model_combo.currentText()
            
            # Update button text for single model mode
            if self.transcription_settings.enabled:
                self.render_button.setText("Extract Audio & Transcribe")
            else:
                self.render_button.setText("Extract Selected Audio")
                
        elif selected_mode == 1:  # Model Comparison
            self.transcription_settings.enable_model_comparison = True
            self.transcription_settings.enable_voting_system = False
            self.transcription_settings.base_model = self.base_model_combo.currentText()
            
            # Get selected comparison models
            selected_models = []
            for model_id, checkbox in self.comparison_model_checks.items():
                if checkbox.isChecked():
                    selected_models.append(model_id)
            self.transcription_settings.comparison_models = selected_models
            
            # Use base model as the primary model
            self.transcription_settings.model_size = self.transcription_settings.base_model
            
            # Update button text for comparison mode
            comparison_count = len(selected_models)
            if comparison_count > 0:
                self.render_button.setText(f"Extract Audio & Compare {comparison_count + 1} Models")
            else:
                self.render_button.setText("Extract Audio & Compare Models")
                
        elif selected_mode == 2:  # Voting System
            self.transcription_settings.enable_model_comparison = False
            self.transcription_settings.enable_voting_system = True
            self.transcription_settings.comparison_models = []
            self.transcription_settings.model_size = "large-v2"  # Default for voting
            
            # Update voting system parameters
            self.transcription_settings.voting_confidence_threshold = self.confidence_spin.value()
            self.transcription_settings.voting_agreement_threshold = self.agreement_spin.value()
            self.transcription_settings.voting_prefer_english = self.prefer_english_check.isChecked()
            self.transcription_settings.voting_english_bias = self.english_bias_spin.value()
            
            # Update button text for voting mode
            model_count = len(self.transcription_settings.voting_models)
            self.render_button.setText(f"Extract Audio & Vote with {model_count} Models")
        
        # Other settings remain the same
        self.transcription_settings.language = self.language_combo.currentText()
        self.transcription_settings.device = self.device_combo.currentText()
        self.transcription_settings.create_srt = self.create_srt_check.isChecked()
        self.transcription_settings.create_combined_srt = self.create_combined_srt_check.isChecked()
        self.transcription_settings.use_improved_alignment = self.improved_alignment_check.isChecked()
        self.transcription_settings.batch_size = self.batch_size_spin.value()
        self.transcription_settings.compute_type = self.compute_type_combo.currentText()
        
        # Log settings change
        if selected_mode == 1:  # Model comparison
            self.log(f"Model comparison enabled: base={self.transcription_settings.base_model}, "
                    f"comparison_models={self.transcription_settings.comparison_models}")
        elif selected_mode == 2:  # Voting system
            self.log(f"Voting system enabled: {len(self.transcription_settings.voting_models)} models, "
                    f"confidence_threshold={self.transcription_settings.voting_confidence_threshold}, "
                    f"agreement_threshold={self.transcription_settings.voting_agreement_threshold}, "
                    f"prefer_english={self.transcription_settings.voting_prefer_english}, "
                    f"english_bias={self.transcription_settings.voting_english_bias}")
        else:  # Single model
            self.log(f"Single model transcription: model={self.transcription_settings.model_size}, "
                    f"language={self.transcription_settings.language}, "
                    f"device={self.transcription_settings.device}")

    def on_audio_processing_finished(self, clips_info):
        """Handle completed audio processing and transcription."""
        if not clips_info:
            self.log("No clips were processed successfully.", error=True)
            self.enable_ui_elements(True)
            self.render_button.setEnabled(True)
            self.browse_button.setEnabled(True)
            self.output_browse_button.setEnabled(True)
            return
            
        # Store the clip info
        self.clips_info = clips_info
        
        # Check transcription mode
        selected_mode = self.mode_button_group.checkedId()
        
        if selected_mode == 1:  # Model Comparison
            if (self.transcription_settings.enabled and 
                self.transcription_settings.enable_model_comparison and 
                self.transcription_settings.comparison_models):
                
                self.log("Starting model comparison...")
                self.start_model_comparison(clips_info)
                return  # Don't finish processing yet, wait for comparison
                
        elif selected_mode == 2:  # Voting System
            if (self.transcription_settings.enabled and 
                self.transcription_settings.enable_voting_system):
                
                self.log("Starting voting system...")
                self.start_model_voting(clips_info)
                return  # Don't finish processing yet, wait for voting
        
        # Normal completion (single model)
        self.complete_processing(clips_info)

    def start_model_voting(self, clips_info):
        """Start model voting processing"""
        self.voting_processor = ModelVotingProcessor(clips_info, self.transcription_settings)
        self.voting_processor.progress_updated.connect(self.update_progress)
        self.voting_processor.voting_progress.connect(lambda msg: self.log(msg))
        self.voting_processor.error_occurred.connect(lambda msg: self.log(msg, error=True))
        self.voting_processor.voting_finished.connect(self.on_model_voting_finished)
        self.voting_processor.start()

    def on_model_voting_finished(self, voting_results):
        """Handle completed model voting"""
        self.log(f"Model voting completed for {len(voting_results)} clips")
        
        if voting_results:
            dialog = ModelVotingDialog(voting_results, self)
            result = dialog.exec_()
            
            if result == dialog.Accepted:
                self.log("Voting system results processed")
                
                # Create final SRT files from voting results if needed
                if (self.transcription_settings.enabled and 
                    self.transcription_settings.create_combined_srt):
                    
                    self.log("Creating combined SRT files from voting results...")
                    self.create_combined_srt_from_voting_results(voting_results)
        
        self.complete_processing(self.clips_info)

    def create_combined_srt_from_voting_results(self, voting_results):
        """Create combined SRT files from voting results"""
        if not voting_results:
            self.log("No voting results to create combined SRT from.")
            return None
        
        try:
            # Get unique track indices
            track_indices = set()
            for result in voting_results:
                track_idx = result.clip_info.get("track_index")
                if track_idx is not None:
                    track_indices.add(track_idx)
            
            if not track_indices:
                self.log("No valid track indices found in voting results.")
                return None
            
            # Create combined SRT for each track
            created_srts = []
            for track_idx in track_indices:
                # Use consistent naming scheme
                if len(track_indices) == 1:
                    combined_path = os.path.join(self.output_dir, "combined_transcription_voted.srt")
                else:
                    combined_path = os.path.join(self.output_dir, f"combined_transcription_voted_{track_idx}.srt")
                
                result = self.create_combined_srt_for_track_from_voting_results(voting_results, track_idx, combined_path)
                if result:
                    created_srts.append((track_idx, combined_path))
            
            if created_srts:
                for track_idx, srt_path in created_srts:
                    self.log(f"Voting system combined SRT - Track {track_idx}: {os.path.basename(srt_path)}")
            
            return created_srts if created_srts else None
            
        except Exception as e:
            self.log(f"Error creating combined SRT from voting results: {str(e)}", error=True)
            import traceback
            self.log(traceback.format_exc(), error=True)
            return None

    def create_combined_srt_for_track_from_voting_results(self, voting_results, track_index, output_path):
        """Create a combined SRT file for a specific track using voting results"""
        if not voting_results:
            self.log(f"No voting results provided for track {track_index}.")
            return None
        
        try:
            # Filter voting results for this track and sort by timeline
            track_results = []
            for result in voting_results:
                if result.clip_info.get("track_index") == track_index:
                    track_results.append(result)
            
            if not track_results:
                self.log(f"No voting results found for track {track_index}")
                return None
            
            # Sort by timeline start
            track_results = sorted(track_results, key=lambda x: x.clip_info.get("timeline_start", 0))
            
            self.log(f"Processing {len(track_results)} voting results for track {track_index} combined SRT")
            
            # Collect all voted segments with timeline offsets
            all_segments = []
            
            for result in track_results:
                if not result.voted_segments:
                    continue
                
                timeline_start = result.clip_info.get("timeline_start", 0)
                
                self.log(f"Processing voted clip: {result.clip_info.get('name', 'Unknown')} with {len(result.voted_segments)} segments")
                
                # Add voted segments with timeline offset
                for segment in result.voted_segments:
                    if segment.text and segment.text.strip():
                        all_segments.append({
                            "start": timeline_start + segment.start,
                            "end": timeline_start + segment.end,
                            "text": segment.text.strip(),
                            "source": "voted"
                        })
            
            if not all_segments:
                self.log(f"No valid voted segments found for track {track_index}")
                return None
            
            # Sort by start time
            all_segments.sort(key=lambda x: x["start"])
            
            # Fix overlaps (same as other methods)
            cleaned_segments = []
            for i, segment in enumerate(all_segments):
                if not segment["text"]:
                    continue
                
                clean_segment = segment.copy()
                
                # Check for overlap with the previous segment
                if cleaned_segments and clean_segment["start"] < cleaned_segments[-1]["end"]:
                    # Adjust timing to prevent overlap
                    if len(clean_segment["text"]) > len(cleaned_segments[-1]["text"]):
                        # Current segment is longer, adjust previous
                        cleaned_segments[-1]["end"] = clean_segment["start"] - 0.01
                    else:
                        # Previous segment is longer, adjust current
                        clean_segment["start"] = cleaned_segments[-1]["end"] + 0.01
                
                # Only add if segment has positive duration
                if clean_segment["end"] > clean_segment["start"]:
                    cleaned_segments.append(clean_segment)
            
            # Apply text formatting for natural flow
            for i, segment in enumerate(cleaned_segments):
                text = segment["text"].strip()
                
                if i == 0:  # First segment
                    if text and text[0].islower():
                        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
                    if text.endswith("."):
                        text = text[:-1]
                elif i == len(cleaned_segments) - 1:  # Last segment
                    if text and text[0].isupper() and i > 0:
                        text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
                    if not text.endswith("."):
                        text = text + "."
                else:  # Middle segments
                    if text and text[0].isupper() and i > 0:
                        text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
                    if text.endswith("."):
                        text = text[:-1]
                
                segment["text"] = text
            
            # Write SRT file
            segment_count = 0
            with open(output_path, 'w', encoding='utf-8') as srt_file:
                for segment in cleaned_segments:
                    segment_count += 1
                    
                    # Write segment number
                    srt_file.write(f"{segment_count}\n")
                    
                    # Write timestamps
                    start_time_str = self.seconds_to_srt_time(segment["start"])
                    end_time_str = self.seconds_to_srt_time(segment["end"])
                    srt_file.write(f"{start_time_str} --> {end_time_str}\n")
                    
                    # Write text
                    srt_file.write(f"{segment['text']}\n\n")
            
            if segment_count > 0:
                self.log(f"✓ Created voting-based combined SRT for track {track_index}:")
                self.log(f"  Total segments: {segment_count}")
                self.log(f"  Output: {os.path.basename(output_path)}")
                return True
            else:
                self.log(f"No segments to write for track {track_index}")
                return None
        
        except Exception as e:
            self.log(f"Error creating voting-based combined SRT for track {track_index}: {str(e)}", error=True)
            import traceback
            self.log(traceback.format_exc(), error=True)
            return None

    def display_selected_audio_cuts(self):
        """Display cuts/scenes for selected audio tracks - UPDATED for tree-based selection."""
        self.audio_cuts_tree.clear()
        
        # Get selected tracks from tree widget
        selected_tracks = []
        
        def collect_selected_tracks(item):
            """Recursively collect selected track items."""
            if item.isSelected():
                item_data = item.data(0, Qt.UserRole)
                if isinstance(item_data, int):  # This is a track item (has track_index)
                    track_index = item_data
                    stream_index = item.data(0, Qt.UserRole + 1)
                    track_info = item.data(0, Qt.UserRole + 2)
                    
                    if track_index is not None and stream_index is not None and track_info:
                        selected_tracks.append((track_index, track_info, stream_index))
            
            # Check children
            for i in range(item.childCount()):
                collect_selected_tracks(item.child(i))
        
        # Process all top-level items (OTIO files)
        for i in range(self.audio_track_list.topLevelItemCount()):
            collect_selected_tracks(self.audio_track_list.topLevelItem(i))
        
        if not selected_tracks:
            return
        
        # Disable updates during population for better performance
        self.audio_cuts_tree.setUpdatesEnabled(False)
        
        try:
            # Global index counter across all files/tracks
            global_index = 1
            total_clips = 0
            
            # First pass: count total clips for progress indication
            for track_index, track_info, stream_index in selected_tracks:
                track = track_info['track']
                clips = track.children if hasattr(track, 'children') else track
                for cut in clips:
                    if self.get_media_path_from_clip(cut):
                        total_clips += 1
            
            # Show progress for large datasets
            if total_clips > 500:
                from PyQt5.QtWidgets import QProgressDialog
                progress_dialog = QProgressDialog(f"Loading {total_clips} clips...", "Cancel", 0, total_clips, self)
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.show()
                QApplication.processEvents()
            else:
                progress_dialog = None
            
            # Group tracks by file for better organization
            tracks_by_file = {}
            for track_index, track_info, stream_index in selected_tracks:
                file_path = track_info['file_path']
                if file_path not in tracks_by_file:
                    tracks_by_file[file_path] = []
                tracks_by_file[file_path].append((track_index, track_info, stream_index))
            
            processed_clips = 0
            
            # Create tree structure organized by file with enhanced visual separation
            for file_index, (file_path, file_tracks) in enumerate(tracks_by_file.items()):
                file_name = os.path.basename(file_path)
                
                # Calculate total cuts in this file
                total_cuts_in_file = 0
                for track_index, track_info, stream_index in file_tracks:
                    track = track_info['track']
                    clips = track.children if hasattr(track, 'children') else track
                    for cut in clips:
                        if self.get_media_path_from_clip(cut):
                            total_cuts_in_file += 1
                
                # Create file parent item
                file_item = QTreeWidgetItem(self.audio_cuts_tree)
                file_item.setText(0, f"📁 FILE {file_index + 1}")
                file_item.setText(1, f"{file_name}")
                file_item.setText(2, f"{len(file_tracks)} tracks")
                file_item.setText(3, f"{total_cuts_in_file} cuts total")
                
                # Store file information
                file_item.setData(0, Qt.UserRole, {
                    "type": "file", 
                    "file_name": file_name,
                    "file_path": file_path,
                    "file_index": file_index
                })
                
                # Enhanced file header styling
                file_font = file_item.font(0)
                file_font.setBold(True)
                file_font.setPointSize(file_font.pointSize() + 1)
                for col in range(4):
                    file_item.setFont(col, file_font)
                
                # Color-coded background for each file
                file_colors = [
                    QColor(230, 240, 255),  # Light blue
                    QColor(240, 255, 230),  # Light green  
                    QColor(255, 240, 230),  # Light orange
                    QColor(240, 230, 255),  # Light purple
                    QColor(255, 255, 230),  # Light yellow
                ]
                file_color = file_colors[file_index % len(file_colors)]
                
                for col in range(4):
                    file_item.setBackground(col, file_color)
                
                file_item.setForeground(0, QColor(60, 60, 120))
                file_item.setForeground(1, QColor(60, 60, 60))
                
                # Add tracks under file
                for track_index, track_info, stream_index in file_tracks:
                    track = track_info['track']
                    timeline = self.timelines[file_path]
                    
                    # Create track item
                    track_item = QTreeWidgetItem(file_item)
                    track_item.setText(0, f"🎵 TRACK")
                    track_item.setText(1, f"{track_info['name']} (Stream {stream_index})")
                    track_item.setText(2, "")
                    track_item.setText(3, "")
                    track_item.setData(0, Qt.UserRole, {
                        "type": "track", 
                        "track_index": track_index,
                        "local_track_index": track_info['local_track_index'],
                        "stream_index": stream_index,
                        "file_path": file_path
                    })
                    
                    # Enhanced track styling
                    track_font = track_item.font(1)
                    track_font.setBold(True)
                    track_item.setFont(0, track_font)
                    track_item.setFont(1, track_font)
                    
                    track_color = file_color.lighter(110)
                    track_item.setBackground(0, track_color)
                    track_item.setBackground(1, track_color)
                    
                    track_item.setForeground(0, QColor(80, 80, 80))
                    track_item.setForeground(1, QColor(40, 40, 40))
                    
                    # Add clips with batch processing for performance
                    clips = track.children if hasattr(track, 'children') else track
                    track_cut_index = 1
                    
                    # Process clips in batches for better performance
                    clip_batch = []
                    batch_size = 50  # Process 50 clips at a time
                    
                    for cut in clips:
                        if not self.get_media_path_from_clip(cut):
                            continue
                        
                        clip_batch.append((cut, track_cut_index, global_index))
                        track_cut_index += 1
                        global_index += 1
                        
                        # Process batch when it reaches batch_size
                        if len(clip_batch) >= batch_size:
                            self.add_clip_batch_to_tree(clip_batch, track_item, track_index, track_info, stream_index, file_path, file_index, file_color)
                            clip_batch = []
                            processed_clips += len(clip_batch) if clip_batch else batch_size
                            
                            # Update progress and process events
                            if progress_dialog:
                                progress_dialog.setValue(processed_clips)
                                QApplication.processEvents()
                                if progress_dialog.wasCanceled():
                                    return
                    
                    # Process remaining clips in batch
                    if clip_batch:
                        self.add_clip_batch_to_tree(clip_batch, track_item, track_index, track_info, stream_index, file_path, file_index, file_color)
                        processed_clips += len(clip_batch)
                        
                        if progress_dialog:
                            progress_dialog.setValue(processed_clips)
                            QApplication.processEvents()
            
            # For large datasets, don't expand everything automatically
            if total_clips > 200:
                # Only expand file headers, not tracks
                for i in range(self.audio_cuts_tree.topLevelItemCount()):
                    file_item = self.audio_cuts_tree.topLevelItem(i)
                    file_item.setExpanded(True)
                    # Don't expand tracks by default for large datasets
                    for j in range(file_item.childCount()):
                        track_item = file_item.child(j)
                        track_item.setExpanded(False)  # Keep tracks collapsed
                
                self.log(f"Loaded {total_clips} clips. Tracks are collapsed for performance. Click to expand individual tracks as needed.")
            else:
                # Expand all for smaller datasets
                self.audio_cuts_tree.expandAll()
            
            # Update the header to show file context
            if len(tracks_by_file) > 1:
                self.audio_cuts_tree.setHeaderLabels([
                    "Index", "Cut/Scene (by OTIO File)", "Start Time", "Duration"
                ])
            else:
                self.audio_cuts_tree.setHeaderLabels([
                    "Index", "Cut/Scene", "Start Time", "Duration"
                ])
            
            # Close progress dialog
            if progress_dialog:
                progress_dialog.close()
            
            self.log(f"Successfully loaded {total_clips} audio clips from {len(tracks_by_file)} OTIO files")
            
        finally:
            # Re-enable updates
            self.audio_cuts_tree.setUpdatesEnabled(True)

    def add_clip_batch_to_tree(self, clip_batch, track_item, track_index, track_info, stream_index, file_path, file_index, file_color):
        """Add a batch of clips to the tree for better performance."""
        for cut, track_cut_index, global_index in clip_batch:
            cut_name = getattr(cut, 'name', f"Cut {track_cut_index}")
            
            cut_item = QTreeWidgetItem(track_item)
            
            # Enhanced cut display with file context
            cut_item.setText(0, f"{global_index:03d}")
            cut_item.setText(1, f"   └─ {cut_name}")
            
            # Store all necessary information
            start_time = cut.source_range.start_time.to_seconds()
            duration = cut.source_range.duration.to_seconds()
            cut_item.setData(0, Qt.UserRole, {
                "type": "cut",
                "track_index": track_index,
                "local_track_index": track_info['local_track_index'],
                "stream_index": stream_index,
                "clip_index": track_cut_index,
                "global_index": global_index,
                "cut": cut,
                "start_time": start_time,
                "duration": duration,
                "file_path": file_path,
                "timeline": self.timelines[file_path],
                "file_index": file_index
            })
            
            cut_item.setText(2, f"{format_timecode(start_time)}")
            cut_item.setText(3, f"{format_timecode(duration)}")
            cut_item.setTextAlignment(0, Qt.AlignCenter)
            
            # Color-code cuts based on file
            cut_base_color = file_color.lighter(120)
            cut_item.setBackground(0, cut_base_color)
            
            # Alternating row colors within each file
            if track_cut_index % 2 == 0:
                alt_color = cut_base_color.lighter(105)
                for col in range(4):
                    cut_item.setBackground(col, alt_color)

    def select_all_cuts(self):
        """Enhanced select all cuts that works with large datasets and collapsed items."""
        self.log("Selecting all audio cuts...")
        
        # Disable updates for performance
        self.audio_cuts_tree.setUpdatesEnabled(False)
        
        try:
            selected_count = 0
            total_count = 0
            
            # Use a more efficient approach that doesn't require items to be visible
            def select_all_in_item(item):
                nonlocal selected_count, total_count
                
                item_data = item.data(0, Qt.UserRole)
                if item_data and item_data.get("type") == "cut":
                    item.setSelected(True)
                    selected_count += 1
                    total_count += 1
                else:
                    total_count += 1
                
                # Recursively process children
                for i in range(item.childCount()):
                    select_all_in_item(item.child(i))
            
            # Process all top-level items
            for i in range(self.audio_cuts_tree.topLevelItemCount()):
                select_all_in_item(self.audio_cuts_tree.topLevelItem(i))
            
            self.log(f"Selected {selected_count} audio cuts out of {total_count} total items")
            
        finally:
            # Re-enable updates
            self.audio_cuts_tree.setUpdatesEnabled(True)
            
            # Update selection count
            self.update_selection_count()

    def deselect_all_cuts(self):
        """Enhanced deselect all cuts that works with large datasets."""
        self.log("Deselecting all audio cuts...")
        
        # More efficient than clearSelection() for large datasets
        self.audio_cuts_tree.setUpdatesEnabled(False)
        
        try:
            def deselect_all_in_item(item):
                item.setSelected(False)
                
                # Recursively process children
                for i in range(item.childCount()):
                    deselect_all_in_item(item.child(i))
            
            # Process all top-level items
            for i in range(self.audio_cuts_tree.topLevelItemCount()):
                deselect_all_in_item(self.audio_cuts_tree.topLevelItem(i))
            
            self.log("Deselected all audio cuts")
            
        finally:
            self.audio_cuts_tree.setUpdatesEnabled(True)
            self.update_selection_count()

    def invert_cut_selection(self):
        """Enhanced invert cut selection that works with large datasets."""
        self.log("Inverting audio cut selection...")
        
        self.audio_cuts_tree.setUpdatesEnabled(False)
        
        try:
            inverted_count = 0
            
            def invert_selection_in_item(item):
                nonlocal inverted_count
                
                item_data = item.data(0, Qt.UserRole)
                if item_data and item_data.get("type") == "cut":
                    item.setSelected(not item.isSelected())
                    if item.isSelected():
                        inverted_count += 1
                
                # Recursively process children
                for i in range(item.childCount()):
                    invert_selection_in_item(item.child(i))
            
            # Process all top-level items
            for i in range(self.audio_cuts_tree.topLevelItemCount()):
                invert_selection_in_item(self.audio_cuts_tree.topLevelItem(i))
            
            self.log(f"Inverted selection - now {inverted_count} cuts selected")
            
        finally:
            self.audio_cuts_tree.setUpdatesEnabled(True)
            self.update_selection_count()

    def create_output_directories_for_otio_files(self):
        """Create separate output directories for each loaded OTIO file."""
        self.otio_output_dirs = {}
        
        try:
            # Create base output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            for file_path in self.otio_paths:
                # Create safe folder name from OTIO filename
                otio_filename = os.path.splitext(os.path.basename(file_path))[0]
                safe_folder_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in otio_filename)
                
                # Ensure unique folder names in case of duplicates
                base_folder_name = safe_folder_name
                counter = 1
                while any(os.path.basename(existing_dir) == safe_folder_name for existing_dir in self.otio_output_dirs.values()):
                    safe_folder_name = f"{base_folder_name}_{counter}"
                    counter += 1
                
                # Create subfolder for this OTIO file
                otio_output_dir = os.path.join(self.output_dir, safe_folder_name)
                os.makedirs(otio_output_dir, exist_ok=True)
                
                # Store the mapping
                self.otio_output_dirs[file_path] = otio_output_dir
                
                self.log(f"Created output directory for {otio_filename}: {safe_folder_name}/")
            
            self.log(f"Created {len(self.otio_output_dirs)} OTIO-specific output directories")
            return True
            
        except Exception as e:
            self.log(f"Error creating OTIO output directories: {str(e)}", error=True)
            return False

    def clean_output_directory_structure(self):
        """Clean the output directory structure including OTIO subfolders."""
        try:
            if os.path.exists(self.output_dir):
                total_files = 0
                
                # Count files in all subdirectories
                for root, dirs, files in os.walk(self.output_dir):
                    total_files += len(files)
                
                if total_files > 0:
                    reply = QMessageBox.question(
                        self, 
                        "Clean Output Directory Structure", 
                        f"The output directory contains {total_files} files across all OTIO subfolders.\n\n"
                        f"Directory: {self.output_dir}\n\n"
                        "Do you want to delete all existing files before processing?",
                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                        QMessageBox.Yes
                    )
                    
                    if reply == QMessageBox.Cancel:
                        self.log("Processing cancelled by user.")
                        return False
                    elif reply == QMessageBox.No:
                        self.log("Keeping existing files in output directory structure.")
                        return True
                    
                    # Delete and recreate the entire structure
                    shutil.rmtree(self.output_dir)
                    self.log(f"Deleted output directory structure with {total_files} files")
                
                # Recreate base directory
                os.makedirs(self.output_dir, exist_ok=True)
                self.log(f"Recreated clean output directory: {self.output_dir}")
            
            return True
            
        except Exception as e:
            self.log(f"Error cleaning output directory structure: {str(e)}", error=True)
            QMessageBox.critical(
                self, 
                "Directory Error", 
                f"Could not clean output directory structure:\n{str(e)}\n\nProcessing cancelled."
            )
            return False

    def start_audio_rendering(self):
        """Start the audio rendering process - FIXED to ensure OTIO organization works."""
        if not self.timelines:
            self.log("No timeline loaded.", error=True)
            return
        
        # Create separate directories for each OTIO file
        if not self.create_output_directories_for_otio_files():
            return
        
        # Clean output directory structure before processing
        if not self.clean_output_directory_structure():
            return
        
        # Switch to progress tab when processing starts
        for i in range(self.main_tabs.count()):
            if "Progress" in self.main_tabs.tabText(i):
                self.main_tabs.setCurrentIndex(i)
                break
        
        self.current_operation_label.setText("Preparing...")
        
        # Get selected cuts with enhanced information for multiple files
        selected_cuts = []
        
        def process_tree_item(item, parent_file_info=None):
            """Recursively process tree items to find selected cuts."""
            item_data = item.data(0, Qt.UserRole)
            
            if item_data and item_data.get("type") == "file":
                # This is a file header, process its children
                file_info = item_data
                for i in range(item.childCount()):
                    process_tree_item(item.child(i), file_info)
            
            elif item_data and item_data.get("type") == "track":
                # This is a track header, process its children (cuts)
                for i in range(item.childCount()):
                    process_tree_item(item.child(i), parent_file_info)
            
            elif item_data and item_data.get("type") == "cut" and item.isSelected():
                # This is a selected cut
                file_path = item_data.get("file_path")
                if file_path and file_path in self.timelines:
                    # Add timeline reference to cut data
                    cut_data = item_data.copy()
                    cut_data["timeline"] = self.timelines[file_path]
                    selected_cuts.append(cut_data)
        
        # Process all top-level items
        for i in range(self.audio_cuts_tree.topLevelItemCount()):
            process_tree_item(self.audio_cuts_tree.topLevelItem(i))
        
        if not selected_cuts:
            self.log("No audio cuts selected.", error=True)
            QMessageBox.warning(self, "No Selection", "Please select at least one audio cut to extract.")
            return
        
        # Sort selected cuts by global index for organized processing
        selected_cuts.sort(key=lambda x: x.get("global_index", 0))
        
        # Group selected cuts by OTIO file for organized processing and logging
        clips_by_otio = {}
        for cut_data in selected_cuts:
            file_path = cut_data.get("file_path")
            if file_path not in clips_by_otio:
                clips_by_otio[file_path] = []
            clips_by_otio[file_path].append(cut_data)
        
        # Enhanced logging with per-file breakdown
        self.log(f"Selected {len(selected_cuts)} audio cuts from {len(clips_by_otio)} OTIO files:")
        for file_path, file_cuts in clips_by_otio.items():
            otio_name = os.path.basename(file_path)
            output_folder = os.path.basename(self.otio_output_dirs[file_path])
            self.log(f"  {otio_name} → {output_folder}/ ({len(file_cuts)} clips)")
            
            # Debug: Show the exact output directory path
            self.log(f"    Output directory: {self.otio_output_dirs[file_path]}")
            
            # Log individual clips for this OTIO file
            for cut_data in file_cuts:
                global_idx = cut_data.get("global_index", "?")
                clip_idx = cut_data.get("clip_index", "?")
                track_idx = cut_data.get("local_track_index", "?")
                start_time = cut_data.get("start_time", 0)
                duration = cut_data.get("duration", 0)
                self.log(f"    Cut #{global_idx}: Track {track_idx}, Clip {clip_idx} - {format_timecode(start_time)} (Duration: {format_timecode(duration)})")
        
        # Get updated transcription settings
        self.update_transcription_settings()
        
        # Reset tracking flags
        self.audio_tracked = False
        
        # Disable UI elements during rendering
        self.render_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.output_browse_button.setEnabled(False)
        self.audio_track_list.setEnabled(False)
        self.main_tabs.setTabEnabled(0, False)
        self.main_tabs.setTabEnabled(1, False)
        self.main_tabs.setTabEnabled(2, False)
        
        # Update status
        self.current_operation_label.setText("Processing audio clips...")
        self.processing_stats_label.setText(f"0 of {len(selected_cuts)} clips processed")
        
        # Process selected cuts
        self.progress_bar.setValue(0)
        if self.transcription_settings.enabled:
            self.log(f"Processing {len(selected_cuts)} selected audio cuts with transcription...")
        else:
            self.log(f"Processing {len(selected_cuts)} selected audio cuts...")
        
        # CRITICAL FIX: Pass otio_output_dirs to AudioClipProcessor constructor
        self.clip_processor = AudioClipProcessor(
            selected_cuts, 
            self.output_dir, 
            max_workers=4,
            transcription_settings=self.transcription_settings,
            vocal_isolation_settings=self.vocal_isolation_settings,
            otio_output_dirs=self.otio_output_dirs  # PASS IT HERE
        )
        
        # Debug: Verify the processor has the mapping
        self.log(f"DEBUG: Processor has {len(self.clip_processor.otio_output_dirs)} OTIO output directories")
        for file_path, output_dir in self.clip_processor.otio_output_dirs.items():
            otio_name = os.path.basename(file_path)
            folder_name = os.path.basename(output_dir)
            self.log(f"  DEBUG: {otio_name} → {folder_name}")
        
        self.clip_processor.progress_updated.connect(self.update_progress)
        self.clip_processor.clip_processed.connect(self.on_clip_processed)
        self.clip_processor.transcription_progress.connect(self.on_transcription_progress)
        self.clip_processor.error_occurred.connect(lambda msg: self.log(msg, error=True))
        self.clip_processor.debug_info.connect(lambda msg: self.log(msg, error=False))
        self.clip_processor.processing_finished.connect(self.on_audio_processing_finished)
        self.clip_processor.model_changed.connect(self.on_model_changed)
        self.clip_processor.start()

    def on_clip_processed(self, info):
        """Enhanced logging with index information for processed clips."""
        global_index = info.get("global_index", "?")
        display_name = info.get("display_name", info.get("name", "Unknown"))
        
        # Create a more descriptive display name if we have index info
        if global_index != "?":
            if display_name == "Unknown" or not display_name:
                display_name = f"Cut #{global_index}"
            else:
                display_name = f"Cut #{global_index} ({display_name})"
        
        if info.get("srt_path"):
            self.log(f"✓ {display_name} → {os.path.basename(info['output_path'])} with transcription")
        else:
            self.log(f"✓ {display_name} → {os.path.basename(info['output_path'])}")

    def toggle_model_comparison(self):
        """Enable/disable model comparison UI elements"""
        enabled = self.enable_model_comparison.isChecked()
        
        # Enable/disable the comparison group
        self.comparison_group.setEnabled(enabled)
        
        # Enable/disable individual checkboxes
        for checkbox in self.comparison_model_checks.values():
            checkbox.setEnabled(enabled)
        
        # Update settings
        self.update_transcription_settings()

    def update_vocal_isolation_settings(self):
        """Update vocal isolation settings from UI controls"""
        self.vocal_isolation_settings.enabled = self.enable_isolation.isChecked()
        self.vocal_isolation_settings.model = self.model_combo.currentText()
        self.vocal_isolation_settings.extract_vocals = self.isolate_vocals_check.isChecked()
        self.vocal_isolation_settings.extract_instruments = self.isolate_instruments_check.isChecked()

    def search_and_update_media_paths(self):
        """Search for video files and update OTIO media paths for all loaded files - FIXED."""
        if not self.timelines or not self.otio_paths:
            QMessageBox.warning(self, "No Timeline", "Please load OTIO file(s) first.")
            return
        
        # Get all unique media files referenced across all timelines
        missing_files = self.get_missing_media_files()
        
        if not missing_files:
            QMessageBox.information(self, "All Files Found", "All media files are already accessible across all OTIO files.")
            self.refresh_media_status()  # Refresh the UI status
            return
        
        # Group missing files by OTIO source for better user understanding
        files_by_otio = {}
        for missing_file in missing_files:
            otio_file = missing_file.get('otio_file', 'Unknown')
            otio_name = os.path.basename(otio_file)
            if otio_name not in files_by_otio:
                files_by_otio[otio_name] = []
            files_by_otio[otio_name].append(missing_file)
        
        self.log(f"Missing media files by OTIO file:")
        for otio_name, files in files_by_otio.items():
            self.log(f"  {otio_name}: {len(files)} missing files")
        
        # Show dialog with missing files
        dialog = MediaSearchDialog(missing_files, self)
        if dialog.exec_() == dialog.Accepted:
            # Get the updated paths from the dialog
            updated_paths = dialog.get_updated_paths()
            
            if updated_paths:
                # Update the OTIO files with new paths
                self.update_otio_media_paths(updated_paths)
                
                # Save the updated OTIO files
                self.save_updated_otio()
                
                # FIXED: Refresh the UI properly
                self.refresh_media_status()
                
                QMessageBox.information(self, "Update Complete", 
                                    f"Updated {len(updated_paths)} media file paths.\n\n"
                                    f"OTIO files have been saved with new paths.\n"
                                    f"Media status has been refreshed.")

    def get_missing_media_files_for_file(self, file_path):
        """Get missing media files for a specific OTIO file."""
        if file_path not in self.timelines:
            return []
        
        timeline = self.timelines[file_path]
        missing_files = []
        found_files = set()
        
        try:
            for track in timeline.tracks:
                clips = track.children if hasattr(track, 'children') else track
                
                for clip in clips:
                    media_path = self.get_media_path_from_clip(clip)
                    if not media_path or media_path in found_files:
                        continue
                    
                    found_files.add(media_path)
                    
                    if not os.path.exists(media_path):
                        # Try relative to OTIO file location
                        otio_dir = os.path.dirname(file_path)
                        filename = os.path.basename(media_path)
                        alt_path = os.path.join(otio_dir, filename)
                        
                        if not os.path.exists(alt_path):
                            missing_files.append({
                                'original_path': media_path,
                                'filename': filename,
                                'new_path': None,
                                'otio_file': file_path
                            })
                        else:
                            # File found, update reference
                            self.update_media_reference_path_in_file(file_path, media_path, alt_path)
        
        except Exception as e:
            self.log(f"Error checking media files for {os.path.basename(file_path)}: {str(e)}", error=True)
            return []
        
        return missing_files

    def update_media_reference_path_in_file(self, file_path, old_path, new_path):
        """Update media reference path in a specific OTIO file."""
        if file_path not in self.timelines:
            return False
        
        timeline = self.timelines[file_path]
        
        try:
            updated = False
            for track in timeline.tracks:
                clips = track.children if hasattr(track, 'children') else track
                
                for clip in clips:
                    # Handle both singular and plural media reference formats
                    if hasattr(clip, 'media_reference') and clip.media_reference:
                        if hasattr(clip.media_reference, 'target_url') and clip.media_reference.target_url == old_path:
                            clip.media_reference.target_url = new_path
                            updated = True
                    
                    elif hasattr(clip, 'media_references') and clip.media_references:
                        for ref_key, media_ref in clip.media_references.items():
                            if hasattr(media_ref, 'target_url') and media_ref.target_url == old_path:
                                media_ref.target_url = new_path
                                updated = True
            
            return updated
            
        except Exception as e:
            self.log(f"Error updating media reference in {os.path.basename(file_path)}: {str(e)}", error=True)
            return False

    def get_missing_media_files(self):
        """Get list of all missing media files from all loaded OTIO files."""
        all_missing = []
        
        for file_path in self.otio_paths:
            file_missing = self.get_missing_media_files_for_file(file_path)
            all_missing.extend(file_missing)
        
        return all_missing

    def show_media_search_dialog(self, missing_files):
        """Show dialog to let user browse for missing media files."""
        dialog = MediaSearchDialog(missing_files, self)
        if dialog.exec_() == dialog.Accepted:
            # Get the updated paths from the dialog
            updated_paths = dialog.get_updated_paths()
            
            if updated_paths:
                # Update the OTIO file with new paths
                self.update_otio_media_paths(updated_paths)
                
                # Save the updated OTIO file
                self.save_updated_otio()
                
                # Refresh the timeline
                self.load_otio_file(self.otio_path)
                
                QMessageBox.information(self, "Update Complete", 
                                    f"Updated {len(updated_paths)} media file paths.\n"
                                    f"OTIO file has been saved with new paths.")

    def update_otio_media_paths(self, path_mappings):
        """Update media paths in ALL loaded OTIO timelines."""
        updated_count = 0
        
        for file_path, timeline in self.timelines.items():
            file_updated_count = 0
            
            try:
                for track in timeline.tracks:
                    clips = track.children if hasattr(track, 'children') else track
                    
                    for clip in clips:
                        # Handle both singular and plural media reference formats
                        if hasattr(clip, 'media_reference') and clip.media_reference:
                            if hasattr(clip.media_reference, 'target_url'):
                                old_path = clip.media_reference.target_url
                                if old_path in path_mappings:
                                    clip.media_reference.target_url = path_mappings[old_path]
                                    file_updated_count += 1
                                    self.log(f"Updated in {os.path.basename(file_path)}: {os.path.basename(old_path)} -> {path_mappings[old_path]}")
                        
                        elif hasattr(clip, 'media_references') and clip.media_references:
                            for ref_key, media_ref in clip.media_references.items():
                                if hasattr(media_ref, 'target_url'):
                                    old_path = media_ref.target_url
                                    if old_path in path_mappings:
                                        media_ref.target_url = path_mappings[old_path]
                                        file_updated_count += 1
                                        self.log(f"Updated in {os.path.basename(file_path)}: {os.path.basename(old_path)} -> {path_mappings[old_path]}")
                
                updated_count += file_updated_count
                self.log(f"Updated {file_updated_count} media references in {os.path.basename(file_path)}")
                
            except Exception as e:
                self.log(f"Error updating media paths in {os.path.basename(file_path)}: {str(e)}", error=True)
        
        self.log(f"Total: Updated {updated_count} media references across {len(self.timelines)} OTIO files")

    def update_media_reference_path(self, old_path, new_path):
        """Update a single media reference path in the timeline."""
        try:
            for track in self.timeline.tracks:
                clips = track.children if hasattr(track, 'children') else track
                
                for clip in clips:
                    # Handle both singular and plural media reference formats
                    if hasattr(clip, 'media_reference') and clip.media_reference:
                        if hasattr(clip.media_reference, 'target_url') and clip.media_reference.target_url == old_path:
                            clip.media_reference.target_url = new_path
                            return True
                    
                    elif hasattr(clip, 'media_references') and clip.media_references:
                        for ref_key, media_ref in clip.media_references.items():
                            if hasattr(media_ref, 'target_url') and media_ref.target_url == old_path:
                                media_ref.target_url = new_path
                                return True
            
            return False
            
        except Exception as e:
            self.log(f"Error updating media reference: {str(e)}", error=True)
            return False

    def save_updated_otio(self):
        """Save ALL updated OTIO files."""
        try:
            saved_count = 0
            
            for file_path, timeline in self.timelines.items():
                # Create backup of original file
                backup_path = file_path + ".backup"
                if not os.path.exists(backup_path):
                    shutil.copy(file_path, backup_path)
                    self.log(f"Created backup: {os.path.basename(backup_path)}")
                
                # Save the updated timeline
                otio.adapters.write_to_file(timeline, file_path)
                self.log(f"Saved updated OTIO file: {os.path.basename(file_path)}")
                saved_count += 1
            
            # Update media status
            self.media_status_label.setText(f"Media paths updated in {saved_count} files")
            self.media_status_label.setStyleSheet("padding: 6px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; color: green; font-size: 11px;")
            
        except Exception as e:
            self.log(f"Error saving OTIO files: {str(e)}", error=True)
            QMessageBox.critical(self, "Save Error", f"Could not save OTIO files: {str(e)}")

    def load_otio_file(self, file_path):
        """Load an OTIO file (backward compatibility method)."""
        # Clear existing data
        self.timelines.clear()
        self.otio_paths.clear()
        
        # Load the single file
        if self.load_single_otio_file(file_path):
            self.update_ui_after_loading_multiple_files()
        else:
            QMessageBox.critical(self, "Error", f"Could not load OTIO file: {file_path}")

    def check_media_files_status(self):
        """Check the status of media files across all loaded OTIO files - ENHANCED."""
        try:
            total_missing = 0
            files_with_missing = 0
            total_files_checked = 0
            
            for file_path in self.otio_paths:
                missing_files = self.get_missing_media_files_for_file(file_path)
                if missing_files:
                    total_missing += len(missing_files)
                    files_with_missing += 1
                
                # Count total files referenced in this OTIO
                file_count = self.get_total_media_files_for_file(file_path)
                total_files_checked += file_count
            
            # Update status with more detailed information
            if total_missing == 0:
                if total_files_checked > 0:
                    self.media_status_label.setText(f"All {total_files_checked} media files found")
                    self.media_status_label.setStyleSheet("padding: 6px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; color: #155724; font-size: 11px;")
                else:
                    self.media_status_label.setText("No media files to check")
                    self.media_status_label.setStyleSheet("padding: 6px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; color: gray; font-size: 11px;")
            else:
                found_files = total_files_checked - total_missing
                self.media_status_label.setText(f"{found_files}/{total_files_checked} media files found ({total_missing} missing from {files_with_missing} OTIO files)")
                self.media_status_label.setStyleSheet("padding: 6px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24; font-size: 11px;")
            
            self.log(f"Media status check: {total_files_checked - total_missing}/{total_files_checked} files found")
            
        except Exception as e:
            self.media_status_label.setText("Error checking media files")
            self.media_status_label.setStyleSheet("padding: 6px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24; font-size: 11px;")
            self.log(f"Error checking media status: {str(e)}", error=True)

    def get_total_media_files_for_file(self, file_path):
        """Get total count of media files referenced in a specific OTIO file."""
        if file_path not in self.timelines:
            return 0
        
        timeline = self.timelines[file_path]
        media_files = set()
        
        try:
            for track in timeline.tracks:
                clips = track.children if hasattr(track, 'children') else track
                
                for clip in clips:
                    media_path = self.get_media_path_from_clip(clip)
                    if media_path:
                        media_files.add(media_path)
            
            return len(media_files)
            
        except Exception as e:
            self.log(f"Error counting media files for {os.path.basename(file_path)}: {str(e)}", error=True)
            return 0

    def browse_otio(self):
        """Open file dialog to browse for OTIO files - UPDATED to support multiple files."""
        # Use getOpenFileNames for multiple file selection
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Open OTIO File(s)", "", "OTIO Files (*.otio);;All Files (*)"
        )
        
        if file_paths:
            # Clear existing data
            self.timelines.clear()
            self.otio_paths.clear()
            self.audio_track_list.clear()
            self.audio_cuts_tree.clear()
            
            # Load all selected files
            successful_loads = 0
            for file_path in file_paths:
                if self.load_single_otio_file(file_path):
                    successful_loads += 1
            
            if successful_loads > 0:
                # Update UI with combined data from all files
                self.update_ui_after_loading_multiple_files()
                self.log(f"Successfully loaded {successful_loads} of {len(file_paths)} OTIO files.")
            else:
                self.log("Failed to load any OTIO files.", error=True)

    def load_single_otio_file(self, file_path):
        """Load a single OTIO file and add it to the collection."""
        try:
            self.log(f"Loading OTIO file: {os.path.basename(file_path)}")
            timeline = otio.adapters.read_from_file(file_path)
            
            # Store in collections
            self.timelines[file_path] = timeline
            self.otio_paths.append(file_path)
            
            # Update backward compatibility variables (use last loaded file)
            self.timeline = timeline
            self.otio_path = file_path
            
            return True
            
        except Exception as e:
            self.log(f"Error loading OTIO file {os.path.basename(file_path)}: {str(e)}", error=True)
            return False

    def update_ui_after_loading_multiple_files(self):
        """Update UI elements after loading multiple OTIO files - ENHANCED."""
        # Update path label
        if len(self.otio_paths) == 1:
            self.otio_path_label.setText(os.path.basename(self.otio_paths[0]))
        else:
            self.otio_path_label.setText(f"{len(self.otio_paths)} OTIO files loaded")
        
        # Enable media search and new buttons
        self.search_media_button.setEnabled(True)
        self.clear_reload_button.setEnabled(True)
        self.refresh_media_button.setEnabled(True)
        
        # Check media files status across all files
        self.check_media_files_status()
        
        # Reset tracking flags
        self.audio_tracked = False
        
        # Populate audio tracks from all loaded files
        self.populate_audio_tracks_from_multiple_files()

    def populate_audio_tracks_from_multiple_files(self):
        """Populate audio tracks from all loaded OTIO files with enhanced track-position-aware auto-mapping."""
        self.audio_track_list.clear()
        
        # Log with updated auto-mapping info
        self.log(f"Populating tracks with track-position-aware auto-mapping {'enabled' if self.auto_mapping_enabled else 'disabled'}")
        
        track_dict = {}
        global_track_index = 0
        
        for file_path, timeline in self.timelines.items():
            file_name = os.path.basename(file_path)
            self.log(f"Processing tracks from: {file_name}")
            
            # Create OTIO file parent item
            otio_item = QTreeWidgetItem(self.audio_track_list)
            otio_item.setText(0, f"📁 {file_name}")
            otio_item.setText(1, "")  # Will be updated with track count
            otio_item.setText(2, "")
            
            # Style OTIO file header
            font = otio_item.font(0)
            font.setBold(True)
            font.setPointSize(font.pointSize() + 1)
            otio_item.setFont(0, font)
            
            # Color-coded background for OTIO files
            otio_item.setBackground(0, QColor(240, 248, 255))
            otio_item.setBackground(1, QColor(240, 248, 255))
            otio_item.setBackground(2, QColor(240, 248, 255))
            
            # Store OTIO file information
            otio_item.setData(0, Qt.UserRole, {
                "type": "otio_file",
                "file_path": file_path,
                "file_name": file_name
            })
            
            # Make OTIO item non-selectable (only tracks should be selectable)
            otio_item.setFlags(Qt.ItemIsEnabled)

            track_count_for_file = 0
            
            for local_track_index, track in enumerate(timeline.tracks):
                track_kind = getattr(track, 'kind', 'Unknown').lower()
                track_name = getattr(track, 'name', f"Track {local_track_index}")
                
                self.log(f"File {file_name}, Track {local_track_index}: name='{track_name}', kind='{track_kind}'")
                
                # Skip non-audio tracks
                if 'audio' not in track_kind.lower() and track_kind.lower() != 'unknown':
                    self.log(f"Skipping track {local_track_index} (not audio): kind='{track_kind}'")
                    continue
                
                # Count clips with media references
                clip_count = 0
                sample_media_path = None
                
                clips = track.children if hasattr(track, 'children') else track
                for item in clips:
                    if self.get_media_path_from_clip(item):
                        clip_count += 1
                        if not sample_media_path:
                            sample_media_path = self.get_media_path_from_clip(item)
                
                self.log(f"Track {local_track_index} ({track_name}) has {clip_count} clips with media")
                
                if clip_count == 0:
                    self.log(f"Skipping empty track {local_track_index}: {track_name}")
                    continue
                
                # Get stream info
                if sample_media_path and os.path.exists(sample_media_path):
                    stream_indexes = self.identify_audio_streams(sample_media_path)
                else:
                    stream_indexes = [(0, "Default")]
                
                # Create track child item under OTIO file
                track_item = QTreeWidgetItem(otio_item)
                track_item.setText(0, f"🎵 {track_name}")
                track_item.setText(1, f"{clip_count}")
                
                # Enhanced stream selection combo box with track-position-aware auto-mapping
                stream_combo = QComboBox()
                for stream_idx, stream_desc in stream_indexes:
                    stream_combo.addItem(f"Stream {stream_idx} ({stream_desc})", stream_idx)
                
                # UPDATED: Check for existing mapping using composite key (media_path, track_position)
                mapping_key = (sample_media_path, local_track_index)
                if mapping_key in self.media_stream_mappings:
                    mapped_stream = self.media_stream_mappings[mapping_key]
                    # Find and set the combo to the mapped stream
                    for i in range(stream_combo.count()):
                        if stream_combo.itemData(i) == mapped_stream:
                            stream_combo.setCurrentIndex(i)
                            self.log(f"Auto-applied stream {mapped_stream} to {track_name} at position {local_track_index} (from existing mapping)")
                            break
                
                # Set the combo box as widget for column 2
                self.audio_track_list.setItemWidget(track_item, 2, stream_combo)
                
                # Enhanced track info with track position
                track_info = {
                    'track': track,
                    'track_index': global_track_index,
                    'local_track_index': local_track_index,  # This is the key for track position
                    'file_path': file_path,
                    'file_name': file_name,
                    'name': track_name,
                    'clip_count': clip_count,
                    'streams': stream_indexes,
                    'media_path': sample_media_path
                }
                
                # Store data in track item
                track_item.setData(0, Qt.UserRole, global_track_index)
                track_item.setData(0, Qt.UserRole + 1, stream_combo.itemData(stream_combo.currentIndex()))  # Initial stream index
                track_item.setData(0, Qt.UserRole + 2, track_info)
                
                track_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                # UPDATED: Enhanced stream combo change handler with track-position-aware auto-mapping
                def on_stream_changed(idx, combo=stream_combo, item=track_item, media_path=sample_media_path, track_pos=local_track_index):
                    selected_stream = combo.itemData(idx)
                    item.setData(0, Qt.UserRole + 1, selected_stream)
                    
                    # Apply track-position-aware auto-mapping if enabled
                    if self.auto_mapping_enabled and media_path:
                        self.apply_stream_mapping_to_same_media(media_path, selected_stream, exclude_item=item)
                    
                    self.display_selected_audio_cuts()
                
                stream_combo.currentIndexChanged.connect(on_stream_changed)
                
                # Style track item
                track_item.setBackground(0, QColor(248, 252, 255))
                
                track_dict[global_track_index] = track_info
                global_track_index += 1
                track_count_for_file += 1
                
                self.log(f"Added track {global_track_index-1}: {file_name} - {track_name} (position {local_track_index}) with {len(stream_indexes)} streams")
            
            # Update OTIO file item with track count
            if track_count_for_file > 0:
                otio_item.setText(1, f"{track_count_for_file} tracks")
                otio_item.setExpanded(True)  # Expand by default to show tracks
            else:
                # Remove OTIO item if no valid tracks
                index = self.audio_track_list.indexOfTopLevelItem(otio_item)
                if index >= 0:
                    self.audio_track_list.takeTopLevelItem(index)
        
        # Enable/disable UI based on available tracks
        self.enable_ui_elements(True)
        self.log(f"Loaded {len(track_dict)} audio tracks from {len(self.timelines)} OTIO files with track-position-aware auto-mapping.")
        
        # Log media file mapping status with updated format
        if self.media_stream_mappings:
            self.log(f"Active track-position mappings: {len(self.media_stream_mappings)} combinations")
            for (media_path, track_pos), stream in self.media_stream_mappings.items():
                self.log(f"  {os.path.basename(media_path)} at track position {track_pos} → Stream {stream}")
        
        if not track_dict:
            self.log("No audio tracks found in any timeline.", error=True)
            QMessageBox.warning(self, "No Audio Tracks", "No audio tracks were found in any of the loaded timelines.")

    def apply_stream_mapping_to_same_media(self, media_path, selected_stream, exclude_item=None):
        """Apply stream selection to tracks with same media file AND same track position across OTIO files"""
        if not media_path or not self.auto_mapping_enabled:
            return
        
        # Get the track position of the item that triggered this mapping
        if not exclude_item:
            return
        
        exclude_track_info = exclude_item.data(0, Qt.UserRole + 2)
        if not exclude_track_info:
            return
        
        trigger_track_position = exclude_track_info.get('local_track_index')
        if trigger_track_position is None:
            return
        
        # Create composite key: (media_path, track_position)
        mapping_key = (media_path, trigger_track_position)
        
        # Store the mapping with track position
        self.media_stream_mappings[mapping_key] = selected_stream
        
        applied_count = 0
        media_filename = os.path.basename(media_path)
        
        self.log(f"Auto-mapping stream {selected_stream} for media file: {media_filename} at track position {trigger_track_position}")
        
        # Find tracks with SAME media file AND SAME track position across different OTIO files
        for i in range(self.audio_track_list.topLevelItemCount()):
            otio_item = self.audio_track_list.topLevelItem(i)
            
            for j in range(otio_item.childCount()):
                track_item = otio_item.child(j)
                
                # Skip the item that triggered this mapping
                if track_item == exclude_item:
                    continue
                
                track_info = track_item.data(0, Qt.UserRole + 2)
                if not track_info:
                    continue
                
                track_media_path = track_info.get('media_path')
                track_position = track_info.get('local_track_index')
                
                # FIXED: Check both media path AND track position
                if (track_media_path == media_path and 
                    track_position == trigger_track_position):
                    
                    # This track uses the same media file AND is in the same track position
                    stream_combo = self.audio_track_list.itemWidget(track_item, 2)
                    if stream_combo:
                        # Find the matching stream index in this combo
                        for k in range(stream_combo.count()):
                            if stream_combo.itemData(k) == selected_stream:
                                if stream_combo.currentIndex() != k:
                                    # Temporarily disconnect signals to avoid recursive calls
                                    stream_combo.blockSignals(True)
                                    stream_combo.setCurrentIndex(k)
                                    track_item.setData(0, Qt.UserRole + 1, selected_stream)
                                    stream_combo.blockSignals(False)
                                    applied_count += 1
                                    
                                    otio_name = track_info.get('file_name', 'Unknown')
                                    track_name = track_info.get('name', 'Unknown')
                                    self.log(f"  → Applied to {otio_name} - {track_name} (track position {track_position})")
                                break
        
        if applied_count > 0:
            self.log(f"✓ Auto-mapped stream {selected_stream} to {applied_count} tracks at position {trigger_track_position} using {media_filename}")
            
            # Show notification to user
            self.show_auto_mapping_notification(media_filename, selected_stream, applied_count, trigger_track_position)
        else:
            self.log(f"ℹ No other tracks found at position {trigger_track_position} using {media_filename}")

    def show_auto_mapping_notification(self, media_filename, stream_index, count, track_position):
        """Show a brief notification about auto-mapping with track position info"""
        from PyQt5.QtCore import QTimer
        
        # Create a temporary notification label
        notification = QLabel(f"🔗 Auto-mapped stream {stream_index} to {count} tracks at position {track_position} using {media_filename}")
        notification.setStyleSheet("""
            QLabel {
                background-color: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
                margin: 5px;
            }
        """)
        notification.setWordWrap(True)
        
        # Add to the main layout temporarily
        main_layout = self.centralWidget().layout()
        if main_layout:
            main_layout.insertWidget(0, notification)
            
            # Remove after 3 seconds
            QTimer.singleShot(3000, lambda: notification.deleteLater())

    def get_media_path_from_clip(self, clip):
        """Helper method to get media path from a clip (handles different OTIO formats)."""
        if hasattr(clip, 'media_reference') and clip.media_reference:
            if hasattr(clip.media_reference, 'target_url'):
                return clip.media_reference.target_url
        elif hasattr(clip, 'media_references') and clip.media_references:
            if 'DEFAULT_MEDIA' in clip.media_references:
                media_ref = clip.media_references['DEFAULT_MEDIA']
            else:
                media_ref = next(iter(clip.media_references.values()))
            
            if hasattr(media_ref, 'target_url'):
                return media_ref.target_url
        
        return None

    def browse_audio_output(self):
        """Open file dialog to set audio output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir
        )
        
        if dir_path:
            self.output_dir = dir_path
            self.output_path_label.setText(os.path.basename(dir_path) or dir_path)
            self.log(f"Output directory set to: {dir_path}")

    def identify_audio_streams(self, media_path):
        """
        Identify audio streams in the media file and return their indexes and descriptions.
        Improved version with better stream detection.
        """
        try:
            cmd = [
                "ffprobe", 
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "a",
                media_path
            ]
            
            self.log(f"Running ffprobe to detect audio streams in: {media_path}")
            
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            data = json.loads(result.stdout)
            
            # Debug output - print the raw stream data
            self.log(f"Found {len(data.get('streams', []))} audio streams")
            
            stream_info = []
            for i, stream in enumerate(data.get("streams", [])):
                # Get the actual index from the stream data
                actual_index = stream.get("index")
                stream_index = i  # This is the 0-based index (for FFmpeg mapping)
                
                channels = stream.get("channels", "?")
                language = stream.get("tags", {}).get("language", "")
                title = stream.get("tags", {}).get("title", "")
                codec = stream.get("codec_name", "unknown")
                sample_rate = stream.get("sample_rate", "?")
                
                # Create a helpful description
                description = f"{codec}, {channels} ch, {sample_rate} Hz"
                if language:
                    description += f", {language}"
                if title:
                    description += f", {title}"
                
                self.log(f"Stream {stream_index} (index {actual_index}): {description}")
                
                # Store both the mapping index (i) and the description
                stream_info.append((stream_index, description))
            
            if not stream_info:
                # If no streams found, return a default one
                self.log("No audio streams detected, using default (0)")
                return [(0, "Default")]
            
            return stream_info
        except Exception as e:
            self.log(f"Error identifying audio streams: {str(e)}", error=True)
            import traceback
            self.log(traceback.format_exc(), error=True)
            return [(0, "Default")]  # Return default if something goes wrong

    def enable_ui_elements(self, enabled):
        """Enable or disable UI elements based on loaded timeline."""
        self.audio_track_list.setEnabled(enabled and self.audio_track_list.topLevelItemCount() > 0)
        self.render_button.setEnabled(enabled and self.audio_track_list.topLevelItemCount() > 0)

    def on_model_changed(self, model_name):
        """Handle model change during fallback process."""
        self.log(f"Switching to fallback model: {model_name}")

    def on_transcription_progress(self, progress_info):
        """Handle transcription progress updates."""
        clip_index = progress_info.get("clip_index")
        status = progress_info.get("status")
        
        if status == "starting":
            self.log(f"Starting transcription for clip {clip_index}...")
        elif status == "completed":
            self.log(f"Transcription completed for clip {clip_index}")
        elif status == "error":
            error = progress_info.get("error", "Unknown error")
            self.log(f"Transcription error for clip {clip_index}: {error}", error=True)

    def create_combined_srt_for_track(self, clips_info, track_index, output_path):
        """Create a combined SRT file for a specific track index"""
        if not clips_info:
            self.log(f"No clips processed, can't create combined SRT for track {track_index}.")
            return None
                
        try:
            # Create a flattened list of clips from the given track
            timeline_clips = []
            for clip in clips_info:
                # Handle both individual clips and lists of clips
                if isinstance(clip, list):
                    for item in clip:
                        if item.get("track_index") == track_index:
                            timeline_clips.append(item)
                else:
                    if clip.get("track_index") == track_index:
                        timeline_clips.append(clip)
                    
            # Now sort the flattened list
            timeline_clips = sorted(timeline_clips, key=lambda x: x.get("timeline_start", 0))
            
            if not timeline_clips:
                self.log(f"No clips found for track {track_index}")
                return None
            
            # Create a clean combined SRT file
            # First, collect and sort all segments by timeline position
            all_segments = []
            
            # Debug - print how many clips we're processing
            self.log(f"Processing {len(timeline_clips)} clips for track {track_index} combined SRT")
            
            # Collect SRT files from all segments
            for clip in timeline_clips:
                srt_path = clip.get("srt_path")
                if not srt_path or not os.path.exists(srt_path):
                    continue
                    
                # Log which SRT we're parsing
                self.log(f"Parsing SRT file for track {track_index}: {os.path.basename(srt_path)}")
                    
                # Get timeline offset for this clip
                timeline_start = clip.get("timeline_start", 0)
                
                # Parse the SRT file
                segments_from_srt = []
                try:
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Very simple SRT parser
                    blocks = content.strip().split('\n\n')
                    for block in blocks:
                        lines = block.split('\n')
                        if len(lines) >= 3:  # Need at least 3 lines (number, times, text)
                            # Parse time stamps
                            time_line = lines[1]
                            times = time_line.split(' --> ')
                            if len(times) == 2:
                                start_time = self.parse_srt_time(times[0])
                                end_time = self.parse_srt_time(times[1])
                                
                                # Get text (may be multiple lines)
                                text = '\n'.join(lines[2:])
                                
                                segments_from_srt.append({
                                    "start": start_time,
                                    "end": end_time,
                                    "text": text
                                })
                    
                    # Add these segments to our collection with timeline offset
                    for segment in segments_from_srt:
                        # Adjust for timeline position
                        all_segments.append({
                            "start": timeline_start + segment["start"],
                            "end": timeline_start + segment["end"],
                            "text": segment["text"]
                        })
                        
                    self.log(f"Parsed {len(segments_from_srt)} segments from {os.path.basename(srt_path)}")
                        
                except Exception as e:
                    self.log(f"Error parsing SRT file {srt_path}: {str(e)}")
            
            # Sort by start time
            all_segments.sort(key=lambda x: x["start"])
            
            # Check for and fix overlaps
            cleaned_segments = []
            
            for i, segment in enumerate(all_segments):
                # Skip empty segments
                if not segment["text"]:
                    continue
                
                # Start with the current segment
                clean_segment = segment.copy()
                
                # Check for overlap with the previous segment
                if cleaned_segments and clean_segment["start"] < cleaned_segments[-1]["end"]:
                    # Decide how to handle the overlap based on which text is longer/more important
                    if len(clean_segment["text"]) > len(cleaned_segments[-1]["text"]):
                        # Current segment is more important, adjust previous segment
                        cleaned_segments[-1]["end"] = clean_segment["start"] - 0.01
                    else:
                        # Previous segment is more important, adjust current segment
                        clean_segment["start"] = cleaned_segments[-1]["end"] + 0.01
                
                # Only add if the segment has a positive duration after adjustments
                if clean_segment["end"] > clean_segment["start"]:
                    cleaned_segments.append(clean_segment)
            
            # Format segments with proper casing and punctuation
            for i, segment in enumerate(cleaned_segments):
                text = segment["text"].strip()
                
                # Apply formatting rules based on position
                if i == 0:  # First segment
                    # Capitalize first letter
                    if text:
                        text = text[0].upper() + text[1:]
                    # Remove any trailing period if present
                    if text and text.endswith("."):
                        text = text[:-1]
                elif i == len(cleaned_segments) - 1:  # Last segment
                    # Ensure lowercase first letter for non-first segment
                    if text:
                        text = text[0].lower() + text[1:]
                    # Ensure period at end
                    if text and not text.endswith("."):
                        text = text + "."
                else:  # Middle segments
                    # Ensure lowercase first letter
                    if text:
                        text = text[0].lower() + text[1:]
                    # Remove any trailing period if present
                    if text and text.endswith("."):
                        text = text[:-1]
                    
                # Update the segment text with formatted version
                segment["text"] = text
            
            # Write the cleaned segments to SRT
            segment_count = 0
            with open(output_path, 'w', encoding='utf-8') as srt_file:
                for segment in cleaned_segments:
                    segment_count += 1
                    
                    # Write segment number
                    srt_file.write(f"{segment_count}\n")
                    
                    # Write timestamps in SRT format
                    start_time_str = seconds_to_srt_time(segment["start"])
                    end_time_str = seconds_to_srt_time(segment["end"])
                    srt_file.write(f"{start_time_str} --> {end_time_str}\n")
                    
                    # Write text - limit to 2 lines max for gaming subtitles
                    text = segment["text"]
                    # Split on sentence boundaries if possible
                    if len(text) > 60:  # If text is long
                        sentences = text.split('. ')
                        if len(sentences) > 1:
                            formatted_text = '.\n'.join(sentences[:-1]) + '.\n' + sentences[-1]
                        else:
                            # If no sentence boundaries, split at a reasonable point
                            mid_point = len(text) // 2
                            # Find the nearest space
                            split_point = text.rfind(' ', 0, mid_point) if mid_point < len(text) else len(text)
                            if split_point > 0:
                                formatted_text = text[:split_point] + '\n' + text[split_point+1:]
                            else:
                                formatted_text = text
                    else:
                        formatted_text = text
                    
                    srt_file.write(f"{formatted_text}\n\n")
            
            if segment_count > 0:
                self.log(f"Created combined SRT for track {track_index} with {segment_count} segments at {output_path}")
                return True
            else:
                self.log(f"No transcription segments found for track {track_index}, combined SRT not created.")
                return None
                
        except Exception as e:
            self.log(f"Error creating combined SRT for track {track_index}: {str(e)}")
            self.log(traceback.format_exc())
            return None

    def clear_log(self):
        """Clear the log text."""
        self.log_text.clear()

    def save_log(self):
        """Save the log to a file."""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "processing_log.txt", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log(f"Log saved to: {file_path}")
            except Exception as e:
                self.log(f"Error saving log: {str(e)}", error=True)

    def apply_log_filter(self, filter_text):
        """Apply filter to log messages."""
        # Store all original log content if not already stored
        if not hasattr(self, '_full_log_content'):
            self._full_log_content = self.log_text.toPlainText()
        
        # Apply filter based on selection
        if filter_text == "All Messages":
            self.log_text.setPlainText(self._full_log_content)
        elif filter_text == "Info Only":
            lines = self._full_log_content.split('\n')
            filtered_lines = [line for line in lines if 'INFO:' in line or line.strip() == '']
            self.log_text.setPlainText('\n'.join(filtered_lines))
        elif filter_text == "Errors Only":
            lines = self._full_log_content.split('\n')
            filtered_lines = [line for line in lines if 'ERROR:' in line or line.strip() == '']
            self.log_text.setPlainText('\n'.join(filtered_lines))
        
        # Scroll to bottom
        self.log_text.ensureCursorVisible()

    def open_output_folder(self):
        """Open the output folder in file explorer."""
        import subprocess
        import platform
        
        try:
            if platform.system() == "Windows":
                os.startfile(self.output_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", self.output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", self.output_dir])
        except Exception as e:
            self.log(f"Error opening output folder: {str(e)}", error=True)

    def refresh_output_files(self):
        """Refresh the list of output files with OTIO-organized structure."""
        self.output_files_list.clear()
        
        try:
            if os.path.exists(self.output_dir):
                # Get OTIO subdirectories
                items = []
                
                for item_name in sorted(os.listdir(self.output_dir)):
                    item_path = os.path.join(self.output_dir, item_name)
                    
                    if os.path.isdir(item_path):
                        # This is an OTIO subfolder
                        audio_files = []
                        srt_files = []
                        other_files = []
                        
                        try:
                            for file_name in sorted(os.listdir(item_path)):
                                if file_name.endswith('.wav'):
                                    audio_files.append(file_name)
                                elif file_name.endswith('.srt'):
                                    srt_files.append(file_name)
                                elif file_name.endswith('.txt'):
                                    other_files.append(file_name)
                        except Exception:
                            continue
                        
                        total_files = len(audio_files) + len(srt_files) + len(other_files)
                        
                        if total_files > 0:
                            # Add folder header with detailed counts
                            folder_item = QListWidgetItem(
                                f"📁 {item_name}/ ({len(audio_files)} audio, {len(srt_files)} SRT, {len(other_files)} other)"
                            )
                            folder_item.setBackground(QColor(240, 248, 255))
                            font = folder_item.font()
                            font.setBold(True)
                            folder_item.setFont(font)
                            folder_item.setFlags(Qt.ItemIsEnabled)  # Make non-selectable
                            self.output_files_list.addItem(folder_item)
                            
                            # Add files organized by type
                            if srt_files:
                                # SRT files first (most important for review)
                                for file_name in srt_files:
                                    file_item = QListWidgetItem(f"  📄 {file_name}")
                                    self.set_file_icon(file_item, file_name)
                                    self.output_files_list.addItem(file_item)
                            
                            if audio_files:
                                # Then audio files
                                for file_name in audio_files:
                                    file_item = QListWidgetItem(f"  🎵 {file_name}")
                                    self.set_file_icon(file_item, file_name)
                                    self.output_files_list.addItem(file_item)
                            
                            if other_files:
                                # Finally other files
                                for file_name in other_files:
                                    file_item = QListWidgetItem(f"  📄 {file_name}")
                                    self.set_file_icon(file_item, file_name)
                                    self.output_files_list.addItem(file_item)
                    
                    elif item_name.endswith(('.wav', '.srt', '.txt')):
                        # File in root directory (shouldn't happen with new organization)
                        root_item = QListWidgetItem(f"📄 {item_name} (root)")
                        self.set_file_icon(root_item, item_name)
                        self.output_files_list.addItem(root_item)
                
                # Add summary if multiple OTIO folders exist
                if len(self.otio_output_dirs) > 1:
                    summary_item = QListWidgetItem(f"📊 Summary: {len(self.otio_output_dirs)} OTIO projects processed")
                    summary_item.setBackground(QColor(220, 255, 220))
                    font = summary_item.font()
                    font.setItalic(True)
                    summary_item.setFont(font)
                    summary_item.setFlags(Qt.ItemIsEnabled)
                    self.output_files_list.insertItem(0, summary_item)
        
        except Exception as e:
            self.log(f"Error refreshing output files: {str(e)}", error=True)

    def set_file_icon(self, item, filename):
        """Set appropriate icon for file type."""
        if filename.endswith('.wav'):
            item.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        elif filename.endswith('.srt'):
            item.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        else:
            item.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))

    def update_progress(self, value):
        """Update progress bar and processing stats."""
        self.progress_bar.setValue(value)
        
        # Update processing stats if we have clip processor
        if hasattr(self, 'clip_processor') and hasattr(self.clip_processor, 'selected_clips'):
            total_clips = len(self.clip_processor.selected_clips)
            processed_clips = int((value / 100) * total_clips)
            self.processing_stats_label.setText(f"{processed_clips} of {total_clips} clips processed")

    def log(self, message, error=False):
        """Add a message to the log with enhanced filtering support."""
        timestamp = time.strftime("%H:%M:%S")
        prefix = "<span style='color:red'>ERROR:</span> " if error else "<span style='color:green'>INFO:</span> "
        formatted_message = f"[{timestamp}] {prefix}{message}"
        
        # Add to log text
        self.log_text.append(formatted_message)
        self.log_text.ensureCursorVisible()
        
        # Update the full log content for filtering
        if hasattr(self, '_full_log_content'):
            self._full_log_content += "\n" + formatted_message
        else:
            self._full_log_content = self.log_text.toPlainText()
        
        # Auto-refresh output files if this is a completion message
        if "complete" in message.lower() or "created" in message.lower():
            QTimer.singleShot(1000, self.refresh_output_files)  # Refresh after 1 second
        
    def closeEvent(self, event):
        """Handle application close event."""
        # Terminate any running threads
        if hasattr(self, 'clip_processor') and self.clip_processor.isRunning():
            self.clip_processor.terminate()
            self.clip_processor.wait()
        
        # Clean up temp directory
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass
        event.accept()        

    def parse_srt_time(self, time_str):
        """Parse SRT time format to seconds"""
        parts = time_str.strip().split(',')
        if len(parts) != 2:
            return 0
        
        time_parts = parts[0].split(':')
        if len(time_parts) != 3:
            return 0
        
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = int(time_parts[2])
        milliseconds = int(parts[1])
        
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    def start_model_comparison(self, clips_info):
        """Start model comparison processing"""
        self.comparison_processor = ModelComparisonProcessor(clips_info, self.transcription_settings)
        self.comparison_processor.progress_updated.connect(self.update_progress)
        self.comparison_processor.comparison_progress.connect(lambda msg: self.log(msg))
        self.comparison_processor.error_occurred.connect(lambda msg: self.log(msg, error=True))
        self.comparison_processor.comparison_finished.connect(self.on_model_comparison_finished)
        self.comparison_processor.start()

    def create_combined_srt_from_selected(self, selected_clips_data):
        """Create combined SRT files from selected transcriptions (model comparison results)"""
        if not selected_clips_data:
            self.log("No selected clips data provided for combined SRT creation.")
            return None
        
        try:
            # Get unique track indices
            track_indices = set()
            for clip in selected_clips_data:
                track_idx = clip.get("track_index")
                if track_idx is not None:
                    track_indices.add(track_idx)
            
            if not track_indices:
                self.log("No valid track indices found in selected clips data.")
                return None
            
            # Create combined SRT for each track
            created_srts = []
            for track_idx in track_indices:
                combined_path = os.path.join(self.output_dir, f"combined_transcription_{track_idx}.srt")
                result = self.create_combined_srt_for_track_from_selected(selected_clips_data, track_idx, combined_path)
                if result:
                    created_srts.append((track_idx, combined_path))
            
            return created_srts if created_srts else None
            
        except Exception as e:
            self.log(f"Error creating combined SRT from selected transcriptions: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None

    def create_combined_srt_for_track_from_selected(self, selected_clips_data, track_index, output_path):
        """Create a combined SRT file for a specific track using selected transcriptions"""
        if not selected_clips_data:
            self.log(f"No clips data provided for track {track_index}.")
            return None
        
        try:
            # Filter clips for this track and sort by timeline
            track_clips = []
            for clip in selected_clips_data:
                if clip.get("track_index") == track_index:
                    track_clips.append(clip)
            
            if not track_clips:
                self.log(f"No clips found for track {track_index}")
                return None
            
            # Sort by timeline start
            track_clips = sorted(track_clips, key=lambda x: x.get("timeline_start", 0))
            
            self.log(f"Processing {len(track_clips)} clips for track {track_index} combined SRT")
            
            # Collect all segments with timeline offsets
            all_segments = []
            
            for clip in track_clips:
                segments = clip.get("segments", [])
                if not segments:
                    continue
                
                timeline_start = clip.get("timeline_start", 0)
                
                self.log(f"Processing clip: {clip.get('name', 'Unknown')} with {len(segments)} segments")
                
                # Add segments with timeline offset
                for segment in segments:
                    if segment.text and segment.text.strip():
                        all_segments.append({
                            "start": timeline_start + segment.start,
                            "end": timeline_start + segment.end,
                            "text": segment.text.strip()
                        })
            
            if not all_segments:
                self.log(f"No valid segments found for track {track_index}")
                return None
            
            # Sort by start time
            all_segments.sort(key=lambda x: x["start"])
            
            # Fix overlaps
            cleaned_segments = []
            for i, segment in enumerate(all_segments):
                if not segment["text"]:
                    continue
                
                clean_segment = segment.copy()
                
                # Check for overlap with the previous segment
                if cleaned_segments and clean_segment["start"] < cleaned_segments[-1]["end"]:
                    # Adjust timing to prevent overlap
                    if len(clean_segment["text"]) > len(cleaned_segments[-1]["text"]):
                        # Current segment is longer, adjust previous
                        cleaned_segments[-1]["end"] = clean_segment["start"] - 0.01
                    else:
                        # Previous segment is longer, adjust current
                        clean_segment["start"] = cleaned_segments[-1]["end"] + 0.01
                
                # Only add if segment has positive duration
                if clean_segment["end"] > clean_segment["start"]:
                    cleaned_segments.append(clean_segment)
            
            # Apply text formatting
            for i, segment in enumerate(cleaned_segments):
                text = segment["text"].strip()
                
                if i == 0:  # First segment
                    if text:
                        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
                    if text.endswith("."):
                        text = text[:-1]
                elif i == len(cleaned_segments) - 1:  # Last segment
                    if text:
                        text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
                    if not text.endswith("."):
                        text = text + "."
                else:  # Middle segments
                    if text:
                        text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
                    if text.endswith("."):
                        text = text[:-1]
                
                segment["text"] = text
            
            # Write SRT file
            segment_count = 0
            with open(output_path, 'w', encoding='utf-8') as srt_file:
                for segment in cleaned_segments:
                    segment_count += 1
                    
                    # Write segment number
                    srt_file.write(f"{segment_count}\n")
                    
                    # Write timestamps
                    start_time_str = seconds_to_srt_time(segment["start"])
                    end_time_str = seconds_to_srt_time(segment["end"])
                    srt_file.write(f"{start_time_str} --> {end_time_str}\n")
                    
                    # Write formatted text
                    text = segment["text"]
                    if len(text) > 60:  # Split long text
                        sentences = text.split('. ')
                        if len(sentences) > 1:
                            formatted_text = '.\n'.join(sentences[:-1]) + '.\n' + sentences[-1]
                        else:
                            # Split at reasonable point
                            mid_point = len(text) // 2
                            split_point = text.rfind(' ', 0, mid_point) if mid_point < len(text) else len(text)
                            if split_point > 0:
                                formatted_text = text[:split_point] + '\n' + text[split_point+1:]
                            else:
                                formatted_text = text
                    else:
                        formatted_text = text
                    
                    srt_file.write(f"{formatted_text}\n\n")
            
            if segment_count > 0:
                self.log(f"Created combined SRT for track {track_index} with {segment_count} selected segments at {output_path}")
                return True
            else:
                self.log(f"No segments to write for track {track_index}")
                return None
        
        except Exception as e:
            self.log(f"Error creating combined SRT for track {track_index}: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None

    def create_combined_srt_per_otio_file(self, processed_clips):
        """Create separate combined SRT files for each OTIO file."""
        if not processed_clips:
            self.log("No clips processed, can't create combined SRTs.")
            return None
        
        # Group clips by OTIO file
        clips_by_otio = {}
        for clip in processed_clips:
            # Handle both individual clips and lists of clips
            if isinstance(clip, list):
                for item in clip:
                    file_path = item.get("file_path")
                    if file_path:
                        if file_path not in clips_by_otio:
                            clips_by_otio[file_path] = []
                        clips_by_otio[file_path].append(item)
            else:
                file_path = clip.get("file_path")
                if file_path:
                    if file_path not in clips_by_otio:
                        clips_by_otio[file_path] = []
                    clips_by_otio[file_path].append(clip)
        
        created_srts = []
        
        # Create combined SRT for each OTIO file
        for file_path, file_clips in clips_by_otio.items():
            otio_name = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = self.otio_output_dirs.get(file_path, self.output_dir)
            
            # Get unique track indices for this OTIO file
            track_indices = set()
            for clip in file_clips:
                track_idx = clip.get("track_index")
                if track_idx is not None:
                    track_indices.add(track_idx)
            
            # Create combined SRT for each track in this OTIO file
            for track_idx in track_indices:
                if len(track_indices) == 1:
                    # Single track in this OTIO file
                    combined_path = os.path.join(output_dir, f"{otio_name}_combined.srt")
                else:
                    # Multiple tracks in this OTIO file
                    combined_path = os.path.join(output_dir, f"{otio_name}_track{track_idx}_combined.srt")
                
                result = self.create_combined_srt_for_track(file_clips, track_idx, combined_path)
                if result:
                    created_srts.append((file_path, track_idx, combined_path))
                    self.log(f"Created combined SRT: {os.path.basename(output_dir)}/{os.path.basename(combined_path)}")
        
        return created_srts if created_srts else None

    def complete_processing(self, clips_info):
        """Complete the processing workflow with UI updates - UPDATED for OTIO organization."""
        # Re-enable UI elements
        self.render_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.output_browse_button.setEnabled(True)
        self.audio_track_list.setEnabled(True)
        
        # Re-enable all tabs
        for i in range(self.main_tabs.count()):
            self.main_tabs.setTabEnabled(i, True)
        
        # Update status
        self.current_operation_label.setText("Processing Complete")
        self.processing_stats_label.setText(f"Successfully processed {len(clips_info)} clips")
        
        # Reset progress bar
        self.progress_bar.setValue(100)
        
        # Refresh output files
        self.refresh_output_files()
        
        # Update selection count
        self.update_selection_count()

        # Success message
        transcription_msg = " with transcription" if self.transcription_settings.enabled else ""
        self.log(f"Audio extraction{transcription_msg} complete! {len(clips_info)} clips saved to organized folders in: {self.output_dir}")
        
        # Generate combined SRT files per OTIO file
        if (self.transcription_settings.enabled and 
            self.transcription_settings.create_combined_srt and 
            not (self.transcription_settings.enable_model_comparison and 
                self.transcription_settings.comparison_models)):
            
            self.log("Creating combined SRTs for each OTIO file...")
            created_srts = self.create_combined_srt_per_otio_file(clips_info)
            if created_srts:
                for file_path, track_idx, srt_path in created_srts:
                    otio_name = os.path.basename(file_path)
                    folder_name = os.path.basename(os.path.dirname(srt_path))
                    self.log(f"Combined SRT for {otio_name}: {folder_name}/{os.path.basename(srt_path)}")
        
        # Re-enable UI elements
        self.enable_ui_elements(True)
        self.render_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.output_browse_button.setEnabled(True)
        
        # Show completion dialog with OTIO organization info
        completion_msg = f"{len(clips_info)} audio clips processed successfully.\n\n"
        completion_msg += f"Organized by OTIO file in: {self.output_dir}\n"
        completion_msg += f"Created {len(self.otio_output_dirs)} separate folders:\n"
        
        for file_path, output_dir in self.otio_output_dirs.items():
            otio_name = os.path.basename(file_path)
            folder_name = os.path.basename(output_dir)
            completion_msg += f"  • {otio_name} → {folder_name}/\n"
        
        if (self.transcription_settings.enabled and 
            self.transcription_settings.enable_model_comparison and 
            self.transcription_settings.comparison_models):
            completion_msg += "\nModel comparison completed! Combined SRT files have been updated with your selected transcriptions."
        
        QMessageBox.information(self, "Processing Complete", completion_msg)

    def on_model_comparison_finished(self, comparison_results):
        """Handle completed model comparison - skip dialog if no differences found"""
        self.log(f"Model comparison completed for {len(comparison_results)} clips")
        
        if not comparison_results:
            self.log("No comparison results available")
            self.complete_processing(self.clips_info)
            return
        
        # Check if there are any meaningful differences to show
        segments_with_differences = 0
        total_segments = 0
        
        for result in comparison_results:
            # Get all segments for this clip
            all_segments = self.combine_all_segments_for_comparison_check(result)
            segment_groups = self.group_segments_by_time_for_comparison_check(all_segments)
            
            for segment_group in segment_groups:
                total_segments += 1
                if TextComparisonUtils.segment_group_has_differences(segment_group):
                    segments_with_differences += 1
        
        self.log(f"Found {segments_with_differences} segments with differences out of {total_segments} total segments")
        
        # If no differences found, skip the dialog
        if segments_with_differences == 0:
            self.log("🎯 All transcriptions are identical across models - no manual selection needed!")
            self.log("Automatically using base model transcriptions for final SRT files...")
            
            # Create final SRT files directly using base model results
            if (self.transcription_settings.enabled and 
                self.transcription_settings.create_combined_srt):
                
                self.log("Creating combined SRT files from base model transcriptions...")
                self.create_combined_srt_per_otio_file(self.clips_info)
            
            # Show completion message
            QMessageBox.information(
                self, 
                "Model Comparison Complete", 
                f"Model comparison completed successfully!\n\n"
                f"✓ All {total_segments} segments were identical across models\n"
                f"✓ No manual selection required\n"
                f"✓ Using base model ({self.transcription_settings.base_model}) transcriptions\n\n"
                f"Combined SRT files have been created automatically."
            )
            
            self.complete_processing(self.clips_info)
            return
        
        # Show the comparison dialog only if there are differences
        self.log(f"Opening comparison editor for {segments_with_differences} segments with differences...")
        dialog = ModelComparisonDialog(comparison_results, self)
        result = dialog.exec_()
        
        if result == dialog.Accepted:
            self.log("Model comparison dialog closed - creating complete transcription...")
            
            if (self.transcription_settings.enabled and 
                self.transcription_settings.create_combined_srt):
                
                self.log("Merging: Selected transcriptions + All remaining base model segments...")
                selected_data = dialog.get_selected_transcriptions_for_combined_srt()
                
                if selected_data:
                    merged_srts = self.create_complete_merged_srt(selected_data, comparison_results)
                    if merged_srts:
                        self.log(f"✓ Created {len(merged_srts)} complete SRT files with ALL segments")
                        for track_idx, srt_path in merged_srts:
                            self.log(f"  Track {track_idx}: {os.path.basename(srt_path)}")
                    else:
                        self.log("❌ Failed to create complete merged SRT files", error=True)
                else:
                    self.log("❌ No selected transcription data available", error=True)
        
        self.complete_processing(self.clips_info)

    def are_texts_essentially_same_for_comparison(self, text1, text2):
        """Check if two texts are essentially the same (for pre-comparison filtering)"""
        if not text1 or not text2:
            return False
        
        # If texts are exactly the same, they're definitely the same
        if text1.strip() == text2.strip():
            return True
            
        import re
        
        def normalize_text(text):
            # Convert to lowercase
            normalized = text.lower().strip()
            
            # Handle common variations
            normalized = re.sub(r'\bOK\b', 'okay', normalized, flags=re.IGNORECASE)
            normalized = re.sub(r'\bok\b', 'okay', normalized, flags=re.IGNORECASE)
            
            # Normalize whitespace but preserve punctuation
            normalized = re.sub(r'\s+', ' ', normalized)
            
            # Remove only leading/trailing punctuation, keep internal punctuation
            normalized = re.sub(r'^[^\w]+|[^\w]+$', '', normalized)
            
            return normalized
        
        norm1 = normalize_text(text1)
        norm2 = normalize_text(text2)
        
        # Check exact match after normalization
        if norm1 == norm2:
            return True
        
        # Check very high similarity (98%+) only for very similar texts
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Only consider 98%+ similarity as "same" for very close matches
        if similarity >= 0.98:
            return True
        
        return False

    def combine_all_segments_for_comparison_check(self, result):
        """Combine segments from all models for difference checking"""
        all_segments = []
        
        # Add base model segments
        for seg in result.base_segments:
            all_segments.append({
                'model': result.base_model,
                'start': seg.start,
                'end': seg.end,
                'text': seg.text.strip(),
                'is_base': True
            })
        
        # Add comparison model segments
        for comp_result in result.comparison_results:
            for seg in comp_result["segments"]:
                all_segments.append({
                    'model': comp_result["model"],
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text.strip(),
                    'is_base': False
                })
        
        return all_segments

    def group_segments_by_time_for_comparison_check(self, all_segments):
        """Group segments that overlap in time for difference checking"""
        if not all_segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(all_segments, key=lambda x: x['start'])
        
        groups = []
        current_group = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            # Check if this segment overlaps with any segment in current group
            overlaps = False
            for group_seg in current_group:
                if (segment['start'] < group_seg['end'] and 
                    segment['end'] > group_seg['start']):
                    overlaps = True
                    break
            
            if overlaps:
                current_group.append(segment)
            else:
                # Start new group if no overlap
                groups.append(current_group)
                current_group = [segment]
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        return groups

    def create_complete_merged_srt(self, selected_clips_data, comparison_results):
        """Create complete merged SRT by combining selected transcriptions with base model segments"""
        if not selected_clips_data:
            self.log("No selected clips data provided for complete SRT merging.")
            return None
        
        try:
            # Get unique track indices
            track_indices = set()
            for clip in selected_clips_data:
                track_idx = clip.get("track_index")
                if track_idx is not None:
                    track_indices.add(track_idx)
            
            if not track_indices:
                self.log("No valid track indices found in selected clips data.")
                return None
            
            # Create complete merged SRT for each track
            merged_srts = []
            for track_idx in track_indices:
                # Use consistent naming scheme
                if len(track_indices) == 1:
                    combined_path = os.path.join(self.output_dir, "combined_transcription.srt")
                else:
                    combined_path = os.path.join(self.output_dir, f"combined_transcription_{track_idx}.srt")
                
                # Create backup of base model version if it exists
                if os.path.exists(combined_path):
                    backup_path = combined_path + ".base_model_backup"
                    if not os.path.exists(backup_path):  # Don't overwrite existing backup
                        try:
                            shutil.copy(combined_path, backup_path)
                            self.log(f"Created backup: {os.path.basename(backup_path)}")
                        except Exception as e:
                            self.log(f"Warning: Could not create backup: {str(e)}", error=False)
                
                # Create the complete merged SRT
                result = self.create_complete_merged_srt_for_track(
                    selected_clips_data, comparison_results, track_idx, combined_path
                )
                if result:
                    merged_srts.append((track_idx, combined_path))
            
            return merged_srts if merged_srts else None
            
        except Exception as e:
            self.log(f"Error creating complete merged SRT: {str(e)}", error=True)
            import traceback
            self.log(traceback.format_exc(), error=True)
            return None

    def create_complete_merged_srt_for_track(self, selected_clips_data, comparison_results, track_index, output_path):
        """Create complete merged SRT for a track by combining selected + base model segments"""
        try:
            self.log(f"Creating complete merged SRT for track {track_index}")
            
            # Step 1: Get ALL base model segments for this track from processed clips
            all_base_clips = []
            for clip in self.clips_info:
                if isinstance(clip, list):
                    for item in clip:
                        if item.get("track_index") == track_index:
                            all_base_clips.append(item)
                else:
                    if clip.get("track_index") == track_index:
                        all_base_clips.append(clip)
            
            if not all_base_clips:
                self.log(f"No base clips found for track {track_index}")
                return None
            
            # Step 2: Create a mapping of clips that were actually compared
            compared_clip_ids = set()
            for comparison_result in comparison_results:
                clip_info = comparison_result.clip_info
                if clip_info.get("track_index") == track_index:
                    # Use a unique identifier for the clip (could be name, timeline_start, etc.)
                    clip_id = f"{clip_info.get('name', '')}_{clip_info.get('timeline_start', 0)}"
                    compared_clip_ids.add(clip_id)
            
            # Step 3: Create a mapping of selected transcriptions by clip
            selected_segments_by_clip = {}
            for clip in selected_clips_data:
                if clip.get("track_index") != track_index:
                    continue
                
                clip_id = f"{clip.get('name', '')}_{clip.get('timeline_start', 0)}"
                selected_segments_by_clip[clip_id] = clip.get("segments", [])
            
            self.log(f"Track {track_index}: Found {len(all_base_clips)} base clips, {len(compared_clip_ids)} compared clips, {len(selected_segments_by_clip)} with selections")
            
            # Step 4: Build the complete timeline segments
            all_timeline_segments = []
            
            for base_clip in all_base_clips:
                clip_id = f"{base_clip.get('name', '')}_{base_clip.get('timeline_start', 0)}"
                timeline_start = base_clip.get("timeline_start", 0)
                
                # Check if this clip was compared and has selections
                if clip_id in selected_segments_by_clip and clip_id in compared_clip_ids:
                    # Use selected segments for this clip
                    selected_segments = selected_segments_by_clip[clip_id]
                    self.log(f"  Using {len(selected_segments)} selected segments for clip: {base_clip.get('name', 'Unknown')}")
                    
                    for segment in selected_segments:
                        all_timeline_segments.append({
                            "start": timeline_start + segment.start,
                            "end": timeline_start + segment.end,
                            "text": segment.text,
                            "source": "selected",
                            "clip_name": base_clip.get('name', 'Unknown')
                        })
                else:
                    # Use base model segments for this clip (not compared or no selection made)
                    base_segments = base_clip.get("segments", [])
                    source_type = "not_compared" if clip_id not in compared_clip_ids else "base_fallback"
                    
                    self.log(f"  Using {len(base_segments)} base segments for clip: {base_clip.get('name', 'Unknown')} ({source_type})")
                    
                    for segment in base_segments:
                        all_timeline_segments.append({
                            "start": timeline_start + segment.start,
                            "end": timeline_start + segment.end,
                            "text": segment.text,
                            "source": source_type,
                            "clip_name": base_clip.get('name', 'Unknown')
                        })
            
            # Step 5: Sort by timeline position
            all_timeline_segments.sort(key=lambda x: x["start"])
            
            self.log(f"Total timeline segments before cleanup: {len(all_timeline_segments)}")
            
            # Step 6: Remove empty segments and handle overlaps
            cleaned_segments = []
            for segment in all_timeline_segments:
                if not segment["text"] or not segment["text"].strip():
                    continue
                
                # Check for overlap with previous segment
                if cleaned_segments and segment["start"] < cleaned_segments[-1]["end"]:
                    prev_seg = cleaned_segments[-1]
                    
                    self.log(f"  Overlap detected: {prev_seg['clip_name']} ({prev_seg['source']}) vs {segment['clip_name']} ({segment['source']})")
                    
                    # Prioritize selected over base/not_compared
                    if segment["source"] == "selected" and prev_seg["source"] in ["base_fallback", "not_compared"]:
                        # Replace previous with selected
                        cleaned_segments[-1] = segment
                        self.log(f"    Replaced {prev_seg['source']} with selected")
                        continue
                    elif prev_seg["source"] == "selected" and segment["source"] in ["base_fallback", "not_compared"]:
                        # Skip this segment, keep selected
                        self.log(f"    Kept selected, skipped {segment['source']}")
                        continue
                    else:
                        # Adjust timing to prevent overlap
                        overlap_duration = prev_seg["end"] - segment["start"]
                        if overlap_duration < 1.0:  # Small overlap, adjust timing
                            if len(segment["text"]) > len(prev_seg["text"]):
                                prev_seg["end"] = segment["start"] - 0.01
                                self.log(f"    Adjusted previous segment end time")
                            else:
                                segment["start"] = prev_seg["end"] + 0.01
                                self.log(f"    Adjusted current segment start time")
                        else:
                            # Large overlap, keep the longer/more important one
                            if segment["source"] == "selected" or len(segment["text"]) > len(prev_seg["text"]):
                                cleaned_segments[-1] = segment
                                self.log(f"    Replaced due to importance/length")
                                continue
                            else:
                                self.log(f"    Skipped due to importance/length")
                                continue
                
                # Only add if segment has positive duration
                if segment["end"] > segment["start"]:
                    cleaned_segments.append(segment)
            
            self.log(f"Segments after cleanup: {len(cleaned_segments)}")
            
            # Step 7: Apply text formatting for natural flow
            for i, segment in enumerate(cleaned_segments):
                text = segment["text"].strip()
                
                if i == 0:  # First segment
                    if text and text[0].islower():
                        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
                    if text.endswith("."):
                        text = text[:-1]
                elif i == len(cleaned_segments) - 1:  # Last segment
                    if text and text[0].isupper() and i > 0:
                        text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
                    if not text.endswith("."):
                        text = text + "."
                else:  # Middle segments
                    if text and text[0].isupper() and i > 0:
                        text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
                    if text.endswith("."):
                        text = text[:-1]
                
                segment["text"] = text
            
            # Step 8: Write the complete SRT file
            segment_count = 0
            selected_count = 0
            base_count = 0
            not_compared_count = 0
            
            with open(output_path, 'w', encoding='utf-8') as srt_file:
                for segment in cleaned_segments:
                    segment_count += 1
                    
                    if segment["source"] == "selected":
                        selected_count += 1
                    elif segment["source"] == "not_compared":
                        not_compared_count += 1
                    else:
                        base_count += 1
                    
                    # Write segment number
                    srt_file.write(f"{segment_count}\n")
                    
                    # Write timestamps
                    start_time_str = self.seconds_to_srt_time(segment["start"])
                    end_time_str = self.seconds_to_srt_time(segment["end"])
                    srt_file.write(f"{start_time_str} --> {end_time_str}\n")
                    
                    # Write text
                    srt_file.write(f"{segment['text']}\n\n")
            
            self.log(f"✓ Created complete merged SRT for track {track_index}:")
            self.log(f"  Total segments: {segment_count}")
            self.log(f"  Selected transcriptions: {selected_count}")
            self.log(f"  Base model segments: {base_count}")
            self.log(f"  Non-compared segments: {not_compared_count}")
            self.log(f"  Output: {os.path.basename(output_path)}")
            
            return True
            
        except Exception as e:
            self.log(f"Error creating complete merged SRT for track {track_index}: {str(e)}", error=True)
            import traceback
            self.log(traceback.format_exc(), error=True)
            return None

    def seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def get_media_files_by_track(self):
        """Get a mapping of media files to tracks that use them"""
        media_to_tracks = {}
        
        for i in range(self.audio_track_list.topLevelItemCount()):
            otio_item = self.audio_track_list.topLevelItem(i)
            
            for j in range(otio_item.childCount()):
                track_item = otio_item.child(j)
                track_info = track_item.data(0, Qt.UserRole + 2)
                
                if not track_info:
                    continue
                
                media_path = track_info.get('media_path')
                if media_path:
                    if media_path not in media_to_tracks:
                        media_to_tracks[media_path] = []
                    
                    media_to_tracks[media_path].append({
                        'otio_file': track_info.get('file_name', 'Unknown'),
                        'track_name': track_info.get('name', 'Unknown'),
                        'current_stream': track_item.data(0, Qt.UserRole + 1),
                        'file_path': track_info.get('file_path', '')
                    })
        
        return media_to_tracks

class MediaMappingsDialog(QDialog):
    """Dialog to show and manage current media file to stream mappings with track position awareness"""
    
    def __init__(self, media_mappings, timelines, parent=None):
        super().__init__(parent)
        self.media_mappings = media_mappings
        self.timelines = timelines
        self.parent_editor = parent
        self.setup_ui()
        self.populate_mappings()
    
    def setup_ui(self):
        self.setWindowTitle("Track-Position-Aware Media Stream Mappings")
        self.setMinimumSize(1000, 700)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Updated instructions
        instructions = QLabel(
            "This shows which audio stream is selected for each media file at specific track positions across all loaded OTIO files. "
            "With track-position-aware auto-mapping, selecting a stream for track position 2 in one OTIO file will automatically apply "
            "the same stream to track position 2 in OTHER OTIO files that use the same media file.\n\n"
            "🔗 Track-position-aware auto-mapping ensures that track 1 in File A maps to track 1 in File B, track 2 to track 2, etc."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("""
            QLabel {
                margin-bottom: 10px; 
                padding: 12px; 
                background-color: #e9ecef; 
                border-radius: 6px; 
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(instructions)
        
        # Statistics summary
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("font-weight: bold; color: #495057; margin-bottom: 10px;")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        # Updated mappings tree
        self.mappings_tree = QTreeWidget()
        self.mappings_tree.setHeaderLabels(["Media File", "Track Position", "Selected Stream", "Used By", "Actions"])
        self.mappings_tree.setColumnWidth(0, 250)
        self.mappings_tree.setColumnWidth(1, 100)
        self.mappings_tree.setColumnWidth(2, 120)
        self.mappings_tree.setColumnWidth(3, 300)
        self.mappings_tree.setColumnWidth(4, 100)
        self.mappings_tree.setAlternatingRowColors(True)
        layout.addWidget(self.mappings_tree)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.clear_mappings_button = QPushButton("Clear All Mappings")
        self.clear_mappings_button.clicked.connect(self.clear_all_mappings)
        self.clear_mappings_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.populate_mappings)
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        controls_layout.addWidget(self.clear_mappings_button)
        controls_layout.addWidget(self.refresh_button)
        controls_layout.addStretch()
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #545b62;
            }
        """)
        controls_layout.addWidget(self.close_button)
        
        layout.addLayout(controls_layout)
    
    def populate_mappings(self):
        self.mappings_tree.clear()
        
        # Update statistics
        total_mappings = len(self.media_mappings)
        total_otio_files = len(self.timelines)
        self.stats_label.setText(f"📊 {total_mappings} track-position mappings across {total_otio_files} OTIO files")
        
        if not self.media_mappings:
            no_mappings_item = QTreeWidgetItem(self.mappings_tree)
            no_mappings_item.setText(0, "No track-position mappings yet")
            no_mappings_item.setText(1, "—")
            no_mappings_item.setText(2, "Select audio streams to create mappings")
            no_mappings_item.setText(3, "Auto-mapping will track your selections by track position")
            
            # Style the placeholder
            for col in range(4):
                no_mappings_item.setForeground(col, QColor(108, 117, 125))
            font = no_mappings_item.font(0)
            font.setItalic(True)
            no_mappings_item.setFont(0, font)
            return
        
        # Get tracks by media file and position
        media_to_tracks = self.parent_editor.get_media_files_by_track_position()
        
        # Group mappings by media file for better display
        mappings_by_file = {}
        for (media_path, track_position), selected_stream in self.media_mappings.items():
            if media_path not in mappings_by_file:
                mappings_by_file[media_path] = []
            mappings_by_file[media_path].append((track_position, selected_stream))
        
        for media_path, position_streams in mappings_by_file.items():
            media_filename = os.path.basename(media_path)
            
            # Create media file header
            media_header = QTreeWidgetItem(self.mappings_tree)
            media_header.setText(0, f"🎬 {media_filename}")
            media_header.setText(1, "—")
            media_header.setText(2, f"{len(position_streams)} positions")
            media_header.setText(3, "Track position mappings:")
            
            # Style media header
            font = media_header.font(0)
            font.setBold(True)
            media_header.setFont(0, font)
            media_header.setBackground(0, QColor(240, 248, 255))
            
            # Add track position mappings as children
            for track_position, selected_stream in sorted(position_streams):
                mapping_key = (media_path, track_position)
                
                position_item = QTreeWidgetItem(media_header)
                position_item.setText(0, f"  Position {track_position}")
                position_item.setText(1, f"Track {track_position}")
                position_item.setText(2, f"Stream {selected_stream}")
                
                # Show which tracks use this mapping
                tracks_info = media_to_tracks.get(mapping_key, [])
                if tracks_info:
                    usage_text = f"{len(tracks_info)} tracks"
                    position_item.setText(3, usage_text)
                    
                    # Check if all tracks are in sync
                    all_in_sync = all(track['current_stream'] == selected_stream for track in tracks_info)
                    
                    if all_in_sync:
                        position_item.setForeground(2, QColor(40, 167, 69))  # Green for all in sync
                        position_item.setText(3, f"✓ {usage_text} (all in sync)")
                    else:
                        position_item.setForeground(2, QColor(220, 53, 69))  # Red for some out of sync
                        position_item.setText(3, f"⚠ {usage_text} (some out of sync)")
                    
                    # Add individual tracks as sub-children
                    for track_info in tracks_info:
                        track_item = QTreeWidgetItem(position_item)
                        track_item.setText(0, f"    🎵 {track_info['track_name']}")
                        track_item.setText(1, f"({track_info['otio_file']})")
                        
                        current_stream = track_info['current_stream']
                        if current_stream == selected_stream:
                            track_item.setText(2, f"✓ Stream {current_stream}")
                            track_item.setForeground(2, QColor(40, 167, 69))
                        else:
                            track_item.setText(2, f"⚠ Stream {current_stream}")
                            track_item.setForeground(2, QColor(220, 53, 69))
                        
                        track_item.setText(3, "In sync" if current_stream == selected_stream else "Out of sync")
                        
                else:
                    position_item.setText(3, "⚠ No tracks found")
                    position_item.setForeground(3, QColor(220, 53, 69))
                
                # Add clear button for this specific mapping
                clear_button = QPushButton("Clear")
                clear_button.setMaximumWidth(70)
                clear_button.setStyleSheet("""
                    QPushButton {
                        background-color: #6c757d;
                        color: white;
                        border: none;
                        padding: 4px 8px;
                        border-radius: 3px;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background-color: #545b62;
                    }
                """)
                clear_button.clicked.connect(lambda checked, key=mapping_key: self.clear_single_mapping(key))
                self.mappings_tree.setItemWidget(position_item, 4, clear_button)
        
        # Expand all for better visibility
        self.mappings_tree.expandAll()

    def get_media_files_by_track_position(self):
        """Get a mapping of media files with track positions to tracks that use them"""
        media_to_tracks = {}
        
        for i in range(self.audio_track_list.topLevelItemCount()):
            otio_item = self.audio_track_list.topLevelItem(i)
            
            for j in range(otio_item.childCount()):
                track_item = otio_item.child(j)
                track_info = track_item.data(0, Qt.UserRole + 2)
                
                if not track_info:
                    continue
                
                media_path = track_info.get('media_path')
                track_position = track_info.get('local_track_index')
                
                if media_path and track_position is not None:
                    # Create composite key
                    key = (media_path, track_position)
                    
                    if key not in media_to_tracks:
                        media_to_tracks[key] = []
                    
                    media_to_tracks[key].append({
                        'otio_file': track_info.get('file_name', 'Unknown'),
                        'track_name': track_info.get('name', 'Unknown'),
                        'current_stream': track_item.data(0, Qt.UserRole + 1),
                        'file_path': track_info.get('file_path', ''),
                        'track_position': track_position
                    })
        
        return media_to_tracks

    def clear_single_mapping(self, mapping_key):
        """Clear mapping for a single media file + track position combination"""
        media_path, track_position = mapping_key
        media_filename = os.path.basename(media_path)
        
        reply = QMessageBox.question(
            self, 
            "Clear Track Position Mapping", 
            f"Clear the stream mapping for '{media_filename}' at track position {track_position}?\n\n"
            "This will remove the auto-mapping for this specific media file and track position combination, "
            "but won't change current track selections.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if mapping_key in self.parent_editor.media_stream_mappings:
                del self.parent_editor.media_stream_mappings[mapping_key]
                self.parent_editor.log(f"Cleared mapping for {media_filename} at track position {track_position}")
                self.populate_mappings()
    
    def clear_all_mappings(self):
        """Clear all media mappings"""
        if not self.media_mappings:
            QMessageBox.information(self, "No Mappings", "There are no mappings to clear.")
            return
            
        reply = QMessageBox.question(
            self, 
            "Clear All Track Position Mappings", 
            f"Are you sure you want to clear all {len(self.media_mappings)} track-position mappings?\n\n"
            "This will remove all auto-mapping relationships, but won't change current track selections.\n"
            "You can recreate mappings by selecting audio streams again.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.parent_editor.media_stream_mappings.clear()
            self.parent_editor.log("Cleared all track-position mappings")
            self.populate_mappings()

class MediaSearchDialog(QDialog):
    """Enhanced dialog for searching and updating missing media files across multiple OTIO files."""

    def __init__(self, missing_files, parent=None):
        super().__init__(parent)
        self.missing_files = missing_files
        self.path_mappings = {}
        self.setup_ui()
        self.populate_files()

    def setup_ui(self):
        """Set up the enhanced dialog UI."""
        self.setWindowTitle("Update Media Paths - Multiple OTIO Files")
        self.setMinimumSize(900, 700)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "The following media files are missing across your loaded OTIO files. "
            "Browse to locate each file in its new location. Files are grouped by source OTIO file:"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # File tree with OTIO grouping
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["OTIO File / Media Path", "Status", "New Path", "Action"])
        self.file_tree.setColumnWidth(0, 350)
        self.file_tree.setColumnWidth(1, 100)
        self.file_tree.setColumnWidth(2, 300)
        self.file_tree.setColumnWidth(3, 100)
        layout.addWidget(self.file_tree)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.browse_all_button = QPushButton("Browse All in Folder...")
        self.browse_all_button.setToolTip("Browse to a folder containing all media files")
        self.browse_all_button.clicked.connect(self.browse_all_in_folder)
        
        self.auto_search_button = QPushButton("Auto-Search Near OTIO Files...")
        self.auto_search_button.setToolTip("Automatically search for media files in the same directories as OTIO files")
        self.auto_search_button.clicked.connect(self.auto_search_near_otio)
        
        button_layout.addWidget(self.browse_all_button)
        button_layout.addWidget(self.auto_search_button)
        button_layout.addStretch()
        
        self.ok_button = QPushButton("Update Paths")
        self.ok_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)

    def populate_files(self):
        """Populate the file tree with missing files grouped by OTIO source."""
        # Group files by OTIO source
        files_by_otio = {}
        for file_info in self.missing_files:
            if not file_info:  # Safety check
                continue
                
            otio_file = file_info.get('otio_file', 'Unknown')
            otio_name = os.path.basename(otio_file)
            if otio_name not in files_by_otio:
                files_by_otio[otio_name] = []
            files_by_otio[otio_name].append(file_info)
        
        # Create tree structure
        for otio_name, files in files_by_otio.items():
            # Create OTIO file parent item
            otio_item = QTreeWidgetItem(self.file_tree)
            otio_item.setText(0, f"📁 {otio_name}")
            otio_item.setText(1, f"{len(files)} missing")
            otio_item.setText(2, "")
            otio_item.setText(3, "")
            
            # Style OTIO header
            font = otio_item.font(0)
            font.setBold(True)
            otio_item.setFont(0, font)
            otio_item.setBackground(0, QColor(240, 248, 255))
            
            # Add missing files as children
            for file_info in files:
                if not file_info or 'original_path' not in file_info:  # Safety check
                    continue
                    
                file_item = QTreeWidgetItem(otio_item)
                file_item.setText(0, f"  {os.path.basename(file_info['original_path'])}")
                file_item.setText(1, "Missing")
                file_item.setText(2, "Not found")
                file_item.setData(0, Qt.UserRole, file_info)
                
                # Add browse button
                browse_button = QPushButton("Browse...")
                browse_button.clicked.connect(lambda checked, item=file_item: self.browse_for_file(item))
                self.file_tree.setItemWidget(file_item, 3, browse_button)
        
        # Expand all items
        self.file_tree.expandAll()

    def browse_for_file(self, item):
        """Browse for a specific missing file."""
        file_info = item.data(0, Qt.UserRole)
        if not file_info or 'filename' not in file_info:
            return
            
        filename = file_info['filename']
        
        # Try to suggest a reasonable starting directory
        start_dir = os.path.dirname(file_info['original_path'])
        if not os.path.exists(start_dir):
            start_dir = os.path.expanduser("~")
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            f"Locate {filename}",
            start_dir,
            "Video Files (*.mp4 *.mov *.avi *.mkv *.mxf *.r3d *.braw);;All Files (*)"
        )
        
        if file_path and os.path.exists(file_path):
            # Update the item
            item.setText(1, "Found")
            item.setText(2, file_path)
            item.setData(1, Qt.UserRole, file_path)
            
            # Store the mapping
            self.path_mappings[file_info['original_path']] = file_path
            
            # Check if all files are found
            self.check_completion()
    
    def browse_all_in_folder(self):
        """Browse to a folder and try to match all missing files."""
        folder_path = QFileDialog.getExistingDirectory(
            self, 
            "Select folder containing media files",
            os.path.expanduser("~")
        )
        
        if not folder_path:
            return
        
        # Get all video files in the selected folder and subfolders
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mxf', '.r3d', '.braw', '.m4v', '.wmv', '.flv'}
        found_files = {}
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if os.path.splitext(file.lower())[1] in video_extensions:
                    file_path = os.path.join(root, file)
                    found_files[file.lower()] = file_path
        
        # Try to match missing files - FIXED to handle tree structure properly
        matched_count = 0
        
        # Iterate through OTIO file headers (top-level items)
        for i in range(self.file_tree.topLevelItemCount()):
            otio_item = self.file_tree.topLevelItem(i)
            
            # Iterate through missing files (children of OTIO items)
            for j in range(otio_item.childCount()):
                file_item = otio_item.child(j)
                file_info = file_item.data(0, Qt.UserRole)
                
                # Check if file_info exists and has the required data
                if file_info and 'filename' in file_info:
                    filename = file_info['filename'].lower()
                    
                    if filename in found_files:
                        # Found a match
                        matched_path = found_files[filename]
                        file_item.setText(1, "Found")
                        file_item.setText(2, matched_path)
                        file_item.setData(1, Qt.UserRole, matched_path)
                        
                        # Store the mapping
                        self.path_mappings[file_info['original_path']] = matched_path
                        matched_count += 1
        
        if matched_count > 0:
            QMessageBox.information(self, "Files Found", f"Found {matched_count} matching files.")
            self.check_completion()
        else:
            QMessageBox.information(self, "No Matches", "No matching files found in the selected folder.")

    def auto_search_near_otio(self):
        """Automatically search for media files in directories near OTIO files."""
        matched_count = 0
        
        # Get unique OTIO directories
        otio_dirs = set()
        for file_info in self.missing_files:
            if not file_info:
                continue
            otio_file = file_info.get('otio_file')
            if otio_file and otio_file != 'Unknown':
                otio_dirs.add(os.path.dirname(otio_file))
        
        # Search for media files in OTIO directories
        found_files = {}
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mxf', '.r3d', '.braw', '.m4v', '.wmv', '.flv'}
        
        for otio_dir in otio_dirs:
            if os.path.exists(otio_dir):
                for file in os.listdir(otio_dir):
                    if os.path.splitext(file.lower())[1] in video_extensions:
                        file_path = os.path.join(otio_dir, file)
                        found_files[file.lower()] = file_path
        
        # Try to match missing files using the corrected method
        self.match_files_from_dict(found_files)
        
        matched_count = len(self.path_mappings)
        if matched_count > 0:
            QMessageBox.information(self, "Auto-Search Complete", 
                                  f"Found {matched_count} files near OTIO file locations.")
            self.check_completion()
        else:
            QMessageBox.information(self, "No Matches", 
                                  "No matching files found in OTIO file directories.")

    def match_files_from_dict(self, found_files):
        """Match missing files against a dictionary of found files."""
        for file_info in self.missing_files:
            if not file_info or 'filename' not in file_info:
                continue
                
            filename = file_info['filename'].lower()
            
            if filename in found_files:
                matched_path = found_files[filename]
                
                # Find the corresponding tree item using the corrected approach
                item = None
                for i in range(self.file_tree.topLevelItemCount()):
                    top_item = self.file_tree.topLevelItem(i)
                    
                    # Check direct children (missing files)
                    for j in range(top_item.childCount()):
                        child = top_item.child(j)
                        stored_info = child.data(0, Qt.UserRole)
                        if stored_info == file_info:
                            item = child
                            break
                    
                    if item:
                        break
                
                if item:
                    item.setText(1, "Found")
                    item.setText(2, matched_path)
                    item.setData(1, Qt.UserRole, matched_path)
                    
                    # Store the mapping
                    self.path_mappings[file_info['original_path']] = matched_path

    def check_completion(self):
        """Check if all files have been found and enable OK button."""
        all_found = True
        total_files = 0
        found_files = 0
        
        # Iterate through the tree structure properly
        for i in range(self.file_tree.topLevelItemCount()):
            otio_item = self.file_tree.topLevelItem(i)
            
            # Check children (actual missing files)
            for j in range(otio_item.childCount()):
                file_item = otio_item.child(j)
                total_files += 1
                
                if file_item.text(1) == "Found":
                    found_files += 1
                else:
                    all_found = False
        
        self.ok_button.setEnabled(len(self.path_mappings) > 0)
        
        if all_found and total_files > 0:
            self.ok_button.setText("Update All Paths")
        else:
            self.ok_button.setText(f"Update {len(self.path_mappings)} of {total_files} Paths")
    
    def get_updated_paths(self):
        """Get the dictionary of path mappings."""
        return self.path_mappings

def main():
    """Main application entry point."""
    # Check if FFmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("FFmpeg is not installed or not in the system path.")
        print("Please install FFmpeg and make sure it's available in your system path.")
        return 1
        
    app = QApplication(sys.argv)
    window = OTIOAudioEditor()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()