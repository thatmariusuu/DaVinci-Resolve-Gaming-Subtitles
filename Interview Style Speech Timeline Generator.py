import tkinter as tk
from tkinter import messagebox, Scale, IntVar, DoubleVar, ttk
import os
from pydub import AudioSegment
from pydub import effects
import opentimelineio as otio
from collections import defaultdict
import datetime
import logging
import math

# Path where the audio and OTIO files will be saved
output_path = r'C:\Temp'
otio_export_path = "C:/temp/timeline.otio"

# Get the DaVinci Resolve instance
project_manager = resolve.GetProjectManager()
project = project_manager.GetCurrentProject()
timeline_dvr = project.GetCurrentTimeline()
media_pool = project.GetMediaPool()

def export_otio_timeline(export_path, export_type):
    logging.info(f"Exporting timeline to {export_path} as {export_type}...")
    timeline_dvr.Export(
        export_path,
        export_type,
        ""
    )
    logging.info("Export completed.")

def setup_logging():
    # Create a unique log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_path, f"speech_detection_log_{timestamp}.txt")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will keep console output too
        ]
    )
    
    logging.info(f"Starting speech detection script, log file: {log_file}")
    return log_file

def merge_speech_segments(segments, threshold=0.1, padding=0.1):
    # Sort segments by their start times
    segments = sorted(segments)
    
    # Add padding to each segment
    padded_segments = []
    for start, end in segments:
        padded_segments.append((max(0, start - padding), end + padding))
    
    # List to store merged segments
    merged_segments = []
    
    # Initialize the first segment to start merging
    if not padded_segments:
        return []
        
    start, end = padded_segments[0]
    
    for current_start, current_end in padded_segments[1:]:
        # If segments are close enough, merge them
        if current_start <= end + threshold:
            end = max(end, current_end)  # Extend the end if the current segment is further
        else:
            # Save the merged segment and start a new one
            merged_segments.append((start, end))
            start, end = current_start, current_end
    
    # Append the last segment
    merged_segments.append((start, end))
    
    return merged_segments

def find_silent_gaps(segments, timeline_duration, min_gap_duration=0.5):
    """Find silent gaps between speech segments that could be filled with other tracks"""
    if not segments:
        return [(0, timeline_duration)]
    
    # Sort segments by start time
    sorted_segments = sorted(segments)
    silent_gaps = []
    
    # Check for gap at the beginning
    if sorted_segments[0][0] > min_gap_duration:
        silent_gaps.append((0, sorted_segments[0][0]))
    
    # Check for gaps between segments
    for i in range(len(sorted_segments) - 1):
        gap_start = sorted_segments[i][1]
        gap_end = sorted_segments[i + 1][0]
        gap_duration = gap_end - gap_start
        
        if gap_duration >= min_gap_duration:
            silent_gaps.append((gap_start, gap_end))
    
    # Check for gap at the end
    if sorted_segments[-1][1] < timeline_duration - min_gap_duration:
        silent_gaps.append((sorted_segments[-1][1], timeline_duration))
    
    return silent_gaps

def compute_enhanced_segments_with_gap_filling(track_segments_dict, timeline_duration, min_gap_duration=0.5):
    """Enhanced segment computation that fills silent gaps with speech from other tracks"""
    logging.info("Computing enhanced segments with gap filling...")
    
    # Sort tracks by index (higher index = higher priority)
    sorted_tracks = sorted(track_segments_dict.keys())
    
    # Initialize result dictionary
    enhanced_segments = {}
    
    # Track which time periods are already occupied
    occupied_periods = []
    
    # Process tracks in priority order (highest to lowest)
    for track_idx in reversed(sorted_tracks):
        current_segments = track_segments_dict[track_idx]
        track_segments = []
        
        logging.info(f"Processing track {track_idx} with {len(current_segments)} original segments")
        
        # First, add all original segments for this track
        for start, end in current_segments:
            # Check if this segment conflicts with higher priority occupied periods
            conflict = False
            for occ_start, occ_end, occ_track in occupied_periods:
                if start < occ_end and end > occ_start:  # Segments overlap
                    conflict = True
                    break
            
            if not conflict:
                track_segments.append((start, end))
                occupied_periods.append((start, end, track_idx))
        
        # Now, look for silent gaps in higher priority tracks where this track has speech
        for higher_track in [t for t in sorted_tracks if t > track_idx]:
            if higher_track in track_segments_dict:
                higher_segments = track_segments_dict[higher_track]
                
                # Find silent gaps in the higher priority track
                silent_gaps = find_silent_gaps(higher_segments, timeline_duration, min_gap_duration)
                
                logging.info(f"Found {len(silent_gaps)} silent gaps in track {higher_track}")
                
                # For each silent gap, check if current track has speech
                for gap_start, gap_end in silent_gaps:
                    gap_duration = gap_end - gap_start
                    
                    # Find speech segments from current track that could fill this gap
                    for speech_start, speech_end in current_segments:
                        # Check if speech segment overlaps with the silent gap
                        overlap_start = max(gap_start, speech_start)
                        overlap_end = min(gap_end, speech_end)
                        
                        if overlap_start < overlap_end:  # There is an overlap
                            overlap_duration = overlap_end - overlap_start
                            
                            # Only use segments that are substantial enough
                            if overlap_duration >= min_gap_duration:
                                # Check if this period is not already occupied by an even higher priority track
                                conflict = False
                                for occ_start, occ_end, occ_track in occupied_periods:
                                    if occ_track > track_idx and overlap_start < occ_end and overlap_end > occ_start:
                                        conflict = True
                                        break
                                
                                if not conflict:
                                    track_segments.append((overlap_start, overlap_end))
                                    occupied_periods.append((overlap_start, overlap_end, track_idx))
                                    logging.info(f"Added gap-filling segment for track {track_idx}: {overlap_start:.2f}s - {overlap_end:.2f}s (filling gap in track {higher_track})")
        
        # Merge overlapping segments and sort
        enhanced_segments[track_idx] = merge_speech_segments(track_segments, threshold=0.1)
        logging.info(f"Track {track_idx} final segments: {len(enhanced_segments[track_idx])}")
    
    return enhanced_segments

def get_audio_tracks():
    track_count = timeline_dvr.GetTrackCount('audio')
    audio_tracks = [timeline_dvr.GetTrackName('audio', i + 1) for i in range(track_count)]
    return audio_tracks

def render_audio_track(track_index):
    # Ensure project and timeline are properly initialized
    if project is None or timeline_dvr is None:
        print("Error: Project or timeline is not properly initialized.")
        return None

    # Set the current timeline
    project.SetCurrentTimeline(timeline_dvr)

    # Define the preset name based on the track index
    preset_name = f"AudioOnly{track_index}"

    # Load the preset render settings
    if not project.LoadRenderPreset(preset_name):
        print(f"Error: Could not load preset '{preset_name}'")
        return None

    print(f"Preset '{preset_name}' loaded successfully.")

    # Find the source media for this track
    source_clips = []
    
    # Try multiple methods to get track and clips
    try:
        # Method 1: Using GetTrackByIndex
        current_track = timeline_dvr.GetTrackByIndex('audio', track_index)
        if current_track:
            for clip_index in range(current_track.GetClipCount()):
                clip = current_track.GetClipByIndex(clip_index)
                if clip and clip.GetMediaPoolItem():
                    media_pool_item = clip.GetMediaPoolItem()
                    source_filename = media_pool_item.GetName()
                    source_clips.append(source_filename)
    except Exception as e:
        print(f"Error getting track by index: {e}")
    
    # If no clips found, try alternative method
    if not source_clips:
        try:
            # Get all audio tracks
            audio_tracks = [timeline_dvr.GetTrackName('audio', i + 1) for i in range(timeline_dvr.GetTrackCount('audio'))]
            
            # Find media in the track
            track_clips = timeline_dvr.GetItemsInTrack('audio', track_index)
            for clip in track_clips:
                if hasattr(clip, 'GetMediaPoolItem') and clip.GetMediaPoolItem():
                    source_filename = clip.GetMediaPoolItem().GetName()
                    source_clips.append(source_filename)
        except Exception as e:
            print(f"Error finding track clips: {e}")
    
    # Create a descriptive output filename
    if source_clips:
        # Use the first source clip's name, remove extension
        base_filename = os.path.splitext(source_clips[0])[0]
        output_filename = f"{base_filename}_track{track_index}"
    else:
        # Fallback to generic naming
        output_filename = f"audio_track_{track_index}"

    # Set the output file path explicitly
    render_settings = {
        "TargetDir": output_path,
        "CustomName": output_filename
    }
    project.SetRenderSettings(render_settings)
    
    # Add the render job and capture the PID
    pid = project.AddRenderJob()
    print(f"Render Job PID: {pid}")

    # Start rendering the job
    project.StartRendering(pid)

    print(f"Rendering audio track {track_index}")

    # Wait for rendering to complete
    while project.IsRenderingInProgress():
        continue

    # Check for both .mp3 and .wav file extensions
    mp3_output_file_path = os.path.join(output_path, f"{output_filename}.mp3")
    wav_output_file_path = os.path.join(output_path, f"{output_filename}.wav")

    # Check if either file exists
    if os.path.exists(mp3_output_file_path):
        return mp3_output_file_path
    elif os.path.exists(wav_output_file_path):
        return wav_output_file_path
    else:
        print(f"Error: Neither .mp3 nor .wav output files were created.")
        return None

def debug_timeline_cuts(timeline, output_file=None):
    # Ensure log directory exists
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Configure logging
    if output_file:
        logging.basicConfig(
            filename=output_file, 
            level=logging.INFO, 
            format='%(message)s',
            filemode='w'  # Overwrite the file each time
        )
    else:
        logging.basicConfig(
            level=logging.INFO, 
            format='%(message)s'
        )
    
    # Header
    logging.info("Timeline Debug Information")
    logging.info("=" * 80)
    logging.info(f"Timeline Name: {timeline.name}")
    
    # Iterate through all tracks
    for track_index, track in enumerate(timeline.tracks, 1):
        logging.info(f"\nTrack {track_index}: {track.name} (Kind: {track.kind})")
        logging.info("-" * 80)
        
        # Iterate through clips in the track
        for clip_index, clip in enumerate(track, 1):
            try:
                # Handle different types of track elements
                if isinstance(clip, otio.schema.Clip):
                    # Get clip range information
                    if hasattr(clip, 'source_range') and clip.source_range:
                        start_time = clip.source_range.start_time
                        duration = clip.source_range.duration
                        
                        # Try to get more information about the clip
                        clip_name = getattr(clip, 'name', 'Unnamed Clip')
                        
                        # Log detailed clip information
                        logging.info(f"Clip {clip_index}:")
                        logging.info(f"  Name: {clip_name}")
                        logging.info(f"  Type: Clip")
                        
                        # Attempt to get media reference details
                        if clip.media_reference:
                            media_name = getattr(clip.media_reference, 'name', 'Unknown Media')
                            logging.info(f"  Media: {media_name}")
                        
                        # Log time information
                        logging.info(f"  Start Time: {start_time}")
                        logging.info(f"  Duration: {duration}")
                        
                        # Convert to seconds if possible
                        if hasattr(start_time, 'rate'):
                            logging.info(f"  Time Rate: {start_time.rate} fps")
                            logging.info(f"  Start (seconds): {start_time.rescaled_to(1).value}")
                            logging.info(f"  Duration (seconds): {duration.rescaled_to(1).value}")
                
                elif isinstance(clip, otio.schema.Gap):
                    # Handle Gap elements
                    logging.info(f"Clip {clip_index}:")
                    logging.info("  Type: Gap")
                    if hasattr(clip, 'source_range') and clip.source_range:
                        start_time = clip.source_range.start_time
                        duration = clip.source_range.duration
                        logging.info(f"  Start Time: {start_time}")
                        logging.info(f"  Duration: {duration}")
                
                else:
                    # Log other types of track elements
                    logging.info(f"Clip {clip_index}:")
                    logging.info(f"  Type: {type(clip).__name__}")
                    logging.info("  Detailed information not available")
            
            except Exception as e:
                logging.info(f"Error processing clip {clip_index}: {e}")
    
    # Footer
    logging.info("\n" + "=" * 80)
    logging.info("End of Timeline Debug Information")
    
    # If logging to file, print the log file path
    if output_file:
        print(f"Debug log saved to: {output_file}")

def modify_timeline_with_enhanced_interview_style(otio_path, track_segments_dict, timeline_duration):
    # Set up logging for debugging
    logging.info(f"Starting enhanced timeline modification from {otio_path}")
    logging.info(f"Tracks to process: {list(track_segments_dict.keys())}")
    
    try:
        # Load the original timeline
        timeline_otio = otio.adapters.read_from_file(otio_path)
        print(f"Loaded OTIO timeline from {otio_path}.")
        
        # Debug: Log original timeline cuts
        debug_log_path_pre = otio_path.replace(".otio", "_pre_modification_debug.log")
        debug_timeline_cuts(timeline_otio, debug_log_path_pre)
        
        # Check for single-track mode
        single_track_mode = len(track_segments_dict.keys()) < 2
        if single_track_mode:
            print("Single-track mode detected - creating a highlight reel of detected segments")
        else:
            print("Multi-track mode detected - creating enhanced interview-style edit with gap filling")
        
        # Create a new timeline for our result
        timeline_name_suffix = "_highlights" if single_track_mode else "_enhanced_interview"
        new_timeline = otio.schema.Timeline(name=timeline_otio.name + timeline_name_suffix)
        
        # Get the frame rate from the original timeline
        frame_rate = timeline_otio.duration().rate if hasattr(timeline_otio, 'duration') else 30
        logging.info(f"Using frame rate: {frame_rate}")
        
        # Get all original video and audio tracks
        original_video_tracks = [track for track in timeline_otio.tracks if track.kind == "Video"]
        original_audio_tracks = {i+1: track for i, track in enumerate([t for t in timeline_otio.tracks if t.kind == "Audio"])}
        
        logging.info(f"Original timeline has {len(original_video_tracks)} video tracks and {len(original_audio_tracks)} audio tracks")
        
        # If no video track, we can't proceed
        if not original_video_tracks:
            logging.error("No video track found in the timeline")
            print("No video track found in the timeline")
            return None
        
        # IMPROVED: If no segments, return None to indicate failure
        if not any(track_segments_dict.values()):
            print("No speech segments detected.")
            logging.info("No speech segments detected - returning None.")
            return None
        
        # Compute enhanced segments with gap filling
        if not single_track_mode:
            enhanced_segments = compute_enhanced_segments_with_gap_filling(
                track_segments_dict, 
                timeline_duration,
                min_gap_duration=0.5  # Minimum 0.5 second gaps to fill
            )
        else:
            enhanced_segments = track_segments_dict
        
        # Process segments - convert seconds to timeline time
        all_segments = []
        logging.info("Converting enhanced segments to timeline time")
        
        for track_idx, segments in enhanced_segments.items():
            logging.info(f"Processing {len(segments)} enhanced segments for track {track_idx}")
            for start_sec, end_sec in segments:
                # Convert seconds to timeline time
                segment_start = otio.opentime.RationalTime(start_sec * frame_rate, frame_rate)
                segment_end = otio.opentime.RationalTime(end_sec * frame_rate, frame_rate)
                all_segments.append((segment_start, segment_end, track_idx))
        
        # Sort segments by start time
        all_segments.sort(key=lambda x: x[0])
        logging.info(f"Sorted {len(all_segments)} total enhanced segments")
        
        # Identify processed and unprocessed tracks
        processed_tracks = set(track_segments_dict.keys())
        all_audio_tracks = set(original_audio_tracks.keys())
        unprocessed_tracks = all_audio_tracks - processed_tracks
        logging.info(f"Processed tracks: {processed_tracks}")
        logging.info(f"Unprocessed tracks: {unprocessed_tracks}")
        
        # Create corresponding tracks in our new timeline
        # Video tracks
        new_video_tracks = []
        for orig_track in original_video_tracks:
            new_track = otio.schema.Track(name=f"{orig_track.name}{timeline_name_suffix}", kind="Video")
            new_timeline.tracks.append(new_track)
            new_video_tracks.append(new_track)
        
        # Audio tracks
        new_audio_tracks = {}
        for track_idx, orig_track in original_audio_tracks.items():
            new_track = otio.schema.Track(name=f"{orig_track.name}{timeline_name_suffix}", kind="Audio")
            new_timeline.tracks.append(new_track)
            new_audio_tracks[track_idx] = new_track
        
        logging.info(f"Created {len(new_video_tracks)} new video tracks and {len(new_audio_tracks)} new audio tracks")
        
        # Process each segment based on the active speaker
        segments_processed = 0
        for segment_start, segment_end, active_track in all_segments:
            # Skip segments that are too short
            if segment_end - segment_start <= otio.opentime.RationalTime(0, frame_rate):
                logging.info(f"Skipping segment that's too short: {segment_start.value/frame_rate}s - {segment_end.value/frame_rate}s")
                continue
            
            try:
                logging.info(f"Processing enhanced segment {segments_processed+1}: {segment_start.value/frame_rate}s - {segment_end.value/frame_rate}s (Active track: {active_track})")
                segments_processed += 1
                
                # Process all video tracks
                for i, orig_video_track in enumerate(original_video_tracks):
                    new_video_track = new_video_tracks[i]
                    
                    # Find video clips that overlap with this segment
                    for clip in orig_video_track:
                        try:
                            clip_start = clip.range_in_parent().start_time
                            clip_end = clip.range_in_parent().end_time_exclusive()
                            
                            # Check if clip overlaps with the current segment
                            if clip_start <= segment_end and clip_end >= segment_start:
                                # Calculate the overlapping portion
                                overlap_start = max(clip_start, segment_start)
                                overlap_end = min(clip_end, segment_end)
                                
                                # Skip if overlap is too small
                                if overlap_end - overlap_start <= otio.opentime.RationalTime(0, frame_rate):
                                    continue
                                
                                logging.info(f"Creating video clip for overlap: {overlap_start.value/frame_rate}s - {overlap_end.value/frame_rate}s")
                                
                                # Create a new video clip for the overlapping portion
                                new_video_clip = clip.deepcopy()
                                
                                # Calculate source offset
                                source_offset = clip.source_range.start_time + (overlap_start - clip_start)
                                
                                # Set the source range for just this segment
                                new_video_clip.source_range = otio.opentime.TimeRange(
                                    start_time=source_offset,
                                    duration=overlap_end - overlap_start
                                )
                                
                                # Add to the new video track
                                new_video_track.append(new_video_clip)
                        except Exception as e:
                            logging.error(f"Error processing video clip: {e}")
                            continue
                
                # Process all audio tracks
                for track_idx, orig_track in original_audio_tracks.items():
                    # Get our new audio track
                    new_audio_track = new_audio_tracks[track_idx]
                    
                    # Determine if we use custom processing or full track
                    if track_idx in processed_tracks:
                        # If this is a selected track, apply enhanced interview-style rules
                        include_audio = False
                        if single_track_mode:
                            include_audio = (track_idx == active_track)
                        else:
                            include_audio = (track_idx == active_track)
                    else:
                        # Unprocessed tracks always include full audio
                        include_audio = True
                    
                    logging.info(f"Processing audio track {track_idx}, include_audio: {include_audio}")
                    
                    # Find audio clips that overlap with this segment
                    for clip in orig_track:
                        try:
                            clip_start = clip.range_in_parent().start_time
                            clip_end = clip.range_in_parent().end_time_exclusive()
                            
                            # Check if clip overlaps with the current segment
                            if clip_start <= segment_end and clip_end >= segment_start:
                                # Calculate the overlapping portion
                                overlap_start = max(clip_start, segment_start)
                                overlap_end = min(clip_end, segment_end)
                                
                                # Skip if overlap is too small
                                if overlap_end - overlap_start <= otio.opentime.RationalTime(0, frame_rate):
                                    continue
                                
                                if include_audio:
                                    logging.info(f"Creating audio clip for track {track_idx}: {overlap_start.value/frame_rate}s - {overlap_end.value/frame_rate}s")
                                    
                                    # Create a new audio clip for the overlapping portion
                                    new_audio_clip = clip.deepcopy()
                                    
                                    # Calculate source offset
                                    source_offset = clip.source_range.start_time + (overlap_start - clip_start)
                                    
                                    # Set the source range for just this segment
                                    new_audio_clip.source_range = otio.opentime.TimeRange(
                                        start_time=source_offset,
                                        duration=overlap_end - overlap_start
                                    )
                                    
                                    # Add to the new audio track
                                    new_audio_track.append(new_audio_clip)
                                else:
                                    logging.info(f"Creating silent gap for track {track_idx}: {overlap_start.value/frame_rate}s - {overlap_end.value/frame_rate}s")
                                    
                                    # For non-active tracks when processing speech tracks, add a silent gap
                                    silent_gap = otio.schema.Gap(
                                        source_range=otio.opentime.TimeRange(
                                            start_time=overlap_start,
                                            duration=overlap_end - overlap_start
                                        )
                                    )
                                    new_audio_track.append(silent_gap)
                        except Exception as e:
                            logging.error(f"Error processing audio clip: {e}", exc_info=True)
                            continue
            except Exception as e:
                logging.error(f"Error processing segment: {e}", exc_info=True)
                continue
        
        # For unprocessed tracks that haven't been added at all, copy the entire track
        for track_idx in unprocessed_tracks:
            logging.info(f"Processing unprocessed track {track_idx}")
            new_audio_track = new_audio_tracks[track_idx]
            # If no clips were added, copy the entire original track
            if len(new_audio_track) == 0:
                orig_track = original_audio_tracks[track_idx]
                logging.info(f"Copying entire unprocessed track {track_idx} ({len(orig_track)} clips)")
                
                for clip_index, clip in enumerate(orig_track):
                    try:
                        new_audio_track.append(clip.deepcopy())
                        logging.info(f"Successfully copied clip {clip_index+1}")
                    except Exception as e:
                        logging.error(f"Error copying clip {clip_index+1} from unprocessed track {track_idx}: {e}", exc_info=True)
        
        # Save the new timeline
        suffix = "_highlights" if single_track_mode else "_enhanced_interview"
        output_path = otio_path.replace(".otio", suffix + ".otio")
        logging.info(f"Saving enhanced timeline to {output_path}")
        try:
            otio.adapters.write_to_file(new_timeline, output_path)
            print(f"Enhanced timeline saved to {output_path}")
            
            # Debug: Log modified timeline cuts
            debug_log_path_post = output_path.replace(".otio", "_post_modification_debug.log")
            debug_timeline_cuts(new_timeline, debug_log_path_post)
            
            return output_path
        except Exception as e:
            logging.error(f"Error saving timeline: {e}", exc_info=True)
            print(f"Error saving timeline: {e}")
            return None
            
    except Exception as e:
        logging.error(f"Error in modify_timeline_with_enhanced_interview_style: {e}", exc_info=True)
        print(f"Error in modify_timeline_with_enhanced_interview_style: {e}")
        return None

def debug_short_segments(segments, min_duration=0.1, log_file=None):
    filtered_segments = []
    removed_segments = []
    
    for start, end in segments:
        duration = end - start
        
        if duration < min_duration:
            removed_segments.append((start, end, duration))
        else:
            filtered_segments.append((start, end))
    
    # Log details about removed segments
    if removed_segments and log_file:
        with open(log_file, 'a') as f:
            f.write("\n--- Short Segments Analysis ---\n")
            f.write(f"Minimum Duration Threshold: {min_duration} seconds\n")
            f.write("Removed Short Segments:\n")
            for start, end, duration in removed_segments:
                f.write(f"Segment: {start}s - {end}s (Duration: {duration:.4f} seconds)\n")
            f.write(f"Total short segments removed: {len(removed_segments)}\n")
            f.write("--------------------------------\n")
    
    return filtered_segments, removed_segments

def get_timeline_duration(otio_path):
    """Get the total duration of the timeline in seconds"""
    try:
        timeline = otio.adapters.read_from_file(otio_path)
        duration_seconds = timeline.duration().value / timeline.duration().rate
        return duration_seconds
    except Exception as e:
        logging.error(f"Error getting timeline duration: {e}")
        return 300  # Default to 5 minutes if we can't determine

def detect_audio_activity(file_path, threshold_db=-40, chunk_length=10, silence_margin=0.3, min_segment_duration=0.1):
    logging.info(f"Detecting audio activity in {file_path} with threshold: {threshold_db}dB")

    try:
        # Determine file type and load accordingly
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
        elif file_ext == '.wav':
            audio = AudioSegment.from_wav(file_path)
        else:
            logging.error(f"Unsupported audio file type: {file_ext}")
            return []

    except Exception as e:
        logging.error(f"Error reading the audio file: {e}")
        return []

    timestamps = []
    
    # Convert to mono for simpler processing
    audio = audio.set_channels(1)
    
    # Process the audio in chunks
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i+chunk_length]
        
        # Skip if chunk is too short
        if len(chunk) < chunk_length / 2:
            continue
            
        # Check if the chunk's volume is above the threshold
        if chunk.dBFS > threshold_db:
            start_time = i / 1000.0  # Convert to seconds
            end_time = (i + len(chunk)) / 1000.0
            timestamps.append((start_time, end_time))
    
    # Merge segments that are close together
    merged_timestamps = merge_speech_segments(timestamps, threshold=silence_margin)
    
    # Filter out very short segments
    log_file = os.path.join(output_path, f"short_segments_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    filtered_timestamps, removed_segments = debug_short_segments(
        merged_timestamps, 
        min_duration=min_segment_duration,
        log_file=log_file
    )
    
    # Log the results
    logging.info(f"Found {len(filtered_timestamps)} speech segments")
    if removed_segments:
        logging.info(f"Removed {len(removed_segments)} very short segments")
    
    return filtered_timestamps

def split_otio_timeline(timeline_path, target_chunk_duration_minutes=15):
    logging.info(f"Splitting timeline {timeline_path} into {target_chunk_duration_minutes}-minute chunks")
    
    # Load the timeline
    timeline = otio.adapters.read_from_file(timeline_path)
    
    # Get timeline duration in seconds
    timeline_duration_seconds = timeline.duration().value / timeline.duration().rate
    
    # Convert target duration to seconds
    target_chunk_duration_seconds = target_chunk_duration_minutes * 60
    
    # Calculate how many chunks we need
    # We use ceiling to ensure we don't exceed the target duration
    num_chunks = math.ceil(timeline_duration_seconds / target_chunk_duration_seconds)
    
    # Recalculate chunk duration to ensure equal chunks
    actual_chunk_duration_seconds = timeline_duration_seconds / num_chunks
    
    logging.info(f"Timeline duration: {timeline_duration_seconds:.2f} seconds")
    logging.info(f"Creating {num_chunks} chunks of approximately {actual_chunk_duration_seconds:.2f} seconds each")
    
    # Frame rate from the original timeline
    frame_rate = timeline.duration().rate
    
    # List to store paths to the generated timeline chunks
    chunk_paths = []
    
    # Create each chunk
    for chunk_index in range(num_chunks):
        # Calculate chunk start and end times in seconds
        chunk_start_seconds = chunk_index * actual_chunk_duration_seconds
        chunk_end_seconds = min((chunk_index + 1) * actual_chunk_duration_seconds, timeline_duration_seconds)
        
        # Convert to timeline time
        chunk_start = otio.opentime.RationalTime(chunk_start_seconds * frame_rate, frame_rate)
        chunk_end = otio.opentime.RationalTime(chunk_end_seconds * frame_rate, frame_rate)
        
        # Create a new timeline for this chunk
        chunk_name = f"{os.path.splitext(os.path.basename(timeline_path))[0]}_chunk{chunk_index+1}"
        chunk_timeline = otio.schema.Timeline(name=chunk_name)
        
        # For each track in the original timeline
        for original_track in timeline.tracks:
            # Create a new track in the chunk timeline
            new_track = otio.schema.Track(name=original_track.name, kind=original_track.kind)
            chunk_timeline.tracks.append(new_track)
            
            # Track the current position in the new track
            current_position = otio.opentime.RationalTime(0, frame_rate)
            
            # For each clip in the original track
            for clip in original_track:
                try:
                    # Get clip range in parent timeline
                    clip_start_in_timeline = clip.range_in_parent().start_time
                    clip_end_in_timeline = clip.range_in_parent().end_time_exclusive()
                    
                    # Check if clip overlaps with our chunk range
                    if clip_start_in_timeline <= chunk_end and clip_end_in_timeline >= chunk_start:
                        # Calculate the overlapping portion
                        overlap_start = max(clip_start_in_timeline, chunk_start)
                        overlap_end = min(clip_end_in_timeline, chunk_end)
                        
                        # Calculate how far into the chunk this clip starts
                        start_offset = overlap_start - chunk_start
                        
                        # If there's a gap before this clip in our chunk, add a gap
                        if start_offset > current_position:
                            gap_duration = start_offset - current_position
                            gap = otio.schema.Gap(
                                source_range=otio.opentime.TimeRange(
                                    start_time=otio.opentime.RationalTime(0, frame_rate),
                                    duration=gap_duration
                                )
                            )
                            new_track.append(gap)
                            current_position = start_offset
                        
                        # Copy the clip
                        new_clip = clip.deepcopy()
                        
                        # Calculate source offset based on the original clip
                        source_offset = clip.source_range.start_time + (overlap_start - clip_start_in_timeline)
                        
                        # Set the source range for just this segment
                        new_clip.source_range = otio.opentime.TimeRange(
                            start_time=source_offset,
                            duration=overlap_end - overlap_start
                        )
                        
                        # Add to the new track
                        new_track.append(new_clip)
                        
                        # Update current position
                        current_position = current_position + (overlap_end - overlap_start)
                
                except Exception as e:
                    logging.error(f"Error processing clip in chunk {chunk_index+1}: {e}")
        
        # Save the chunk timeline
        chunk_path = os.path.splitext(timeline_path)[0] + f"_chunk{chunk_index+1}.otio"
        otio.adapters.write_to_file(chunk_timeline, chunk_path)
        chunk_paths.append(chunk_path)
        
        logging.info(f"Created chunk {chunk_index+1}/{num_chunks}: {chunk_path}")
    
    return chunk_paths

class AudioRenderGUI:
    def toggle_freq_controls(self, *args):
        """Show or hide frequency controls based on detection method selection"""
        if self.detection_method_var.get() == "voice_isolation":
            self.freq_frame.pack(fill="x", pady=5)
        else:
            self.freq_frame.pack_forget()
    
    def toggle_chunk_controls(self):
        """Show or hide chunk duration controls based on split checkbox selection"""
        if self.split_timeline_var.get():
            self.chunk_frame.pack(fill="x", pady=5)
        else:
            self.chunk_frame.pack_forget()
    
    def toggle_gap_filling_controls(self):
        """Show or hide gap filling controls based on enhanced mode selection"""
        if self.enhanced_mode_var.get():
            self.gap_frame.pack(fill="x", pady=5)
        else:
            self.gap_frame.pack_forget()
            
    def show_help(self):
        """Show help information dialog"""
        help_text = """
Enhanced Speech Detection Help:

Detection Methods:
- Standard: Uses audio energy levels to detect speech
- Voice Isolation: Filters audio to focus on voice frequencies
- Multi-approach: Combines multiple detection methods for best results

Enhanced Interview Mode:
- When enabled, analyzes silent gaps in higher priority tracks
- Fills these gaps with speech from lower priority tracks
- Creates more dynamic, natural-sounding interview edits
- Higher track numbers = higher priority (Track 3 > Track 2 > Track 1)

Sensitivity Settings:
- Audio Sensitivity: Higher values (closer to -20dB) make detection more sensitive
  • Use higher values (-30 to -20) for quiet speakers
  • Use lower values (-60 to -40) for loud environments with background noise

- Silence Margin: Determines how close speech segments must be to merge
  • Higher values (1-2 sec) will merge speech segments with longer pauses
  • Lower values (0.1-0.5 sec) will create more distinct speech segments

- Minimum Gap Duration: Minimum length of silent gaps to fill (Enhanced Mode)
  • Smaller values (0.3-0.5 sec) fill more gaps but may create choppy audio
  • Larger values (1-2 sec) only fill substantial gaps

Advanced Options:
- Voice Frequency Range: Customizes the frequency range for voice isolation
  • Default range (300Hz-3000Hz) works for most human speech
  • Use narrower ranges for cleaner isolation but potential missed speech

- "Normalize audio" option:
  • Helps with consistently quiet audio by boosting volume before detection

- "Single Track Mode" option:
  • Creates a highlight reel from one track instead of interview-style edit
  • Useful when you only want to keep parts where specific audio is detected

- "Enhanced Interview Mode" option:
  • Enables intelligent gap filling between tracks
  • Creates more natural conversation flow
  • Respects track priority (higher numbers = higher priority)

- "Split timeline into chunks" option:
  • Divides the final timeline into multiple equal-sized chunks
  • Specify the target duration for each chunk using the slider

Tips:
- For best interview-style results, enable Enhanced Mode and select multiple tracks
- Track priority matters: Track 3 will override Track 2 which overrides Track 1
- Start with Multi-approach detection for best results
- If results include too much background noise, lower the sensitivity
- If speech is being missed, increase sensitivity or try Voice Isolation
- Use Enhanced Mode with 0.5-1.0 second minimum gap duration for natural flow
- If no speech is detected, try increasing sensitivity or normalizing audio first
        """
        messagebox.showinfo("Enhanced Speech Detection Help", help_text)
            
    def update_progress(self, message, value=None):
        """Update the progress label and progress bar"""
        self.progress_label.config(text=message)
        if value is not None:
            self.progress_var.set(value)
        self.root.update()

    def isolate_voice_and_detect(self, file_path):
        try:
            # Get frequency range from UI
            low_freq = self.low_freq_var.get()
            high_freq = self.high_freq_var.get()
            
            self.update_progress(f"Isolating voice frequencies ({low_freq}Hz-{high_freq}Hz)...", 30)
            logging.info(f"Isolating voice frequencies ({low_freq}Hz-{high_freq}Hz)...")
            
            # Determine file type and load accordingly
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.mp3':
                audio = AudioSegment.from_mp3(file_path)
            elif file_ext == '.wav':
                audio = AudioSegment.from_wav(file_path)
            else:
                logging.error(f"Unsupported audio file type: {file_ext}")
                return []
            
            # Normalize audio if requested
            if self.normalize_audio_var.get():
                self.update_progress("Normalizing audio volume...", 40)
                logging.info("Normalizing audio volume...")
                audio = effects.normalize(audio)
            
            # Convert to mono for simpler processing
            audio = audio.set_channels(1)
            
            # Apply band-pass filter to isolate voice frequencies
            self.update_progress("Applying band-pass filter...", 50)
            logging.info("Applying band-pass filter...")
            filtered_audio = audio.high_pass_filter(low_freq).low_pass_filter(high_freq)
            
            # Save the filtered audio temporarily
            # Use the same extension as the original file
            temp_file = file_path.replace(file_ext, '_filtered.wav')
            filtered_audio.export(temp_file, format="wav")
            
            # Use regular detection on filtered audio
            self.update_progress("Detecting speech in filtered audio...", 60)
            logging.info("Detecting speech in filtered audio...")
            threshold_db = self.sensitivity_var.get()
            segments = detect_audio_activity(temp_file, threshold_db=threshold_db)
            
            # Clean up temporary file
            os.remove(temp_file)
            
            # If no speech detected, try with more sensitive settings
            if not segments:
                self.update_progress("No speech detected. Trying more sensitive settings...", 70)
                logging.info("No speech detected. Trying more sensitive settings...")
                return detect_audio_activity(file_path, threshold_db=threshold_db-10)  # More sensitive threshold
            
            return segments
            
        except Exception as e:
            logging.error(f"Error in voice isolation and detection: {e}")
            # Fall back to regular detection
            return detect_audio_activity(file_path)

    def process_selected_tracks(self):
        # Setup logging at the beginning
        log_file = setup_logging()
        self.update_progress(f"Starting processing... Log file: {log_file}")
        
        # Get selected track indices (1-based)
        selected_tracks = [i+1 for i, var in enumerate(self.check_vars) if var.get()]
        
        if not selected_tracks:
            logging.warning("No tracks selected")
            messagebox.showwarning("Warning", "Please select at least one audio track to process.")
            return
            
        # Check if we're using single track mode or if enough tracks are selected
        single_track_mode = self.single_track_mode_var.get()
        enhanced_mode = self.enhanced_mode_var.get()
        
        if len(selected_tracks) < 2 and not single_track_mode:
            # Ask the user if they want to switch to single track mode
            logging.info("Only one track selected, prompting for single track mode")
            response = messagebox.askyesno(
                "Single Track Mode", 
                "You've selected only one track. Would you like to switch to single track mode?\n\n"
                "This will create a highlights reel of detected speech in this track."
            )
            if response:
                self.single_track_mode_var.set(True)
                logging.info("User switched to single track mode")
            else:
                logging.warning("User declined single track mode with only one track selected")
                messagebox.showwarning("Warning", "Please select at least two audio tracks for interview-style processing or enable single track mode.")
                return
        
        # Reset progress
        self.progress_var.set(0)
        self.update_progress("Starting processing...")
        
        try:
            # Get sensitivity settings
            threshold_db = self.sensitivity_var.get()
            silence_margin = self.silence_margin_var.get()
            min_gap_duration = self.min_gap_var.get() if enhanced_mode else 0.8
            
            logging.info(f"Processing settings: threshold={threshold_db}dB, silence_margin={silence_margin}s")
            logging.info(f"Enhanced mode: {enhanced_mode}, min_gap_duration: {min_gap_duration}s")
            logging.info(f"Selected tracks: {selected_tracks}")
            
            # Export the current timeline as OTIO
            self.update_progress("Exporting timeline to OTIO...", 10)
            export_type = 15.0  # Export type for OTIO
            export_otio_timeline(otio_export_path, export_type)
            
            # Get timeline duration
            timeline_duration = get_timeline_duration(otio_export_path)
            logging.info(f"Timeline duration: {timeline_duration} seconds")
            
            # Dictionary to store speech segments per track
            track_segments = {}
            total_tracks = len(selected_tracks)
            
            # Process each selected track
            for i, track_index in enumerate(selected_tracks):
                track_progress = 20 + (i / total_tracks) * 60
                self.update_progress(f"Processing Audio Track {track_index}...", int(track_progress))
                logging.info(f"Processing Audio Track {track_index}...")
                
                # Render the audio track
                audio_file_path = render_audio_track(track_index)
                if audio_file_path is None:
                    logging.error(f"Failed to render track {track_index}")
                    messagebox.showerror("Error", f"Failed to render track {track_index}. Check DaVinci Resolve render presets.")
                    continue
                
                # Detect speech using the selected method
                detection_method = self.detection_method_var.get()
                detection_progress = int(track_progress + 5)
                logging.info(f"Using detection method: {detection_method}")
                
                if detection_method == "standard":
                    self.update_progress(f"Detecting speech in Track {track_index} (Threshold: {threshold_db}dB)...", detection_progress)
                    segments = detect_audio_activity(
                        audio_file_path, 
                        threshold_db=threshold_db, 
                        silence_margin=silence_margin
                    )
                elif detection_method == "voice_isolation":
                    self.update_progress(f"Isolating voice in Track {track_index}...", detection_progress)
                    segments = self.isolate_voice_and_detect(audio_file_path)
                else:  # multi-approach
                    self.update_progress(f"Running multi-approach detection on Track {track_index}...", detection_progress)
                    segments = self.multi_approach_speech_detection(audio_file_path)
                
                # Store the segments for this track
                track_segments[track_index] = segments
                
                result_progress = int(track_progress + 10)
                self.update_progress(f"Found {len(segments)} speech segments in Track {track_index}", result_progress)
                logging.info(f"Found {len(segments)} speech segments in Track {track_index}")
            
            # If no speech detected in any track, provide helpful feedback
            if not any(track_segments.values()):
                self.update_progress("No speech detected in any tracks", 0)
                logging.warning("No speech detected in any tracks")
                
                # Provide helpful suggestions
                suggestion_message = """No speech detected in any of the selected tracks.

Try these solutions:

1. Increase Audio Sensitivity (move slider towards -20dB)
2. Try "Voice Isolation" detection method
3. Enable "Normalize audio" option for quiet recordings  
4. Check that audio tracks contain actual speech
5. Reduce "Silence Margin" to catch shorter speech segments

Current settings:
• Audio Sensitivity: {threshold_db}dB
• Detection Method: {method}
• Silence Margin: {margin}s""".format(
                    threshold_db=threshold_db,
                    method=self.detection_method_var.get().title(),
                    margin=silence_margin
                )
                
                messagebox.showinfo("No Speech Detected", suggestion_message)
                return

            # Create the timeline with appropriate method
            mode = "highlight reel" if single_track_mode else ("enhanced interview-style timeline" if enhanced_mode else "interview-style timeline")
            self.update_progress(f"Creating {mode}...", 85)
            logging.info(f"Creating {mode}...")
            
            if enhanced_mode and not single_track_mode:
                output_path = modify_timeline_with_enhanced_interview_style(otio_export_path, track_segments, timeline_duration)
            else:
                output_path = modify_timeline_with_interview_style(otio_export_path, track_segments)
            
            if output_path:
                # Check if we need to split the timeline
                if self.split_timeline_var.get():
                    self.update_progress("Splitting timeline into chunks...", 95)
                    logging.info("Splitting timeline into chunks...")
                    try:
                        chunk_duration = self.chunk_duration_var.get()
                        chunk_paths = split_otio_timeline(output_path, target_chunk_duration_minutes=chunk_duration)
                        
                        # Update the success message
                        self.update_progress("Timeline created and split successfully!", 100)
                        logging.info(f"Timeline split into {len(chunk_paths)} chunks")
                        
                        chunk_list = "\n".join(os.path.basename(path) for path in chunk_paths)
                        messagebox.showinfo("Success", 
                            f"{mode.title()} created and split into {len(chunk_paths)} chunks:\n\n"
                            f"{chunk_list}\n\n"
                            f"Original timeline: {os.path.basename(output_path)}\n\n"
                            f"Log file: {log_file}")
                    except Exception as e:
                        logging.error(f"Error splitting timeline: {e}", exc_info=True)
                        self.update_progress(f"Timeline created but splitting failed: {e}", 100)
                        messagebox.showinfo("Partial Success", 
                            f"{mode.title()} created at:\n{output_path}\n\n"
                            f"Timeline splitting failed: {e}\n\n"
                            f"Log file: {log_file}")
                else:
                    self.update_progress("Timeline created successfully!", 100)
                    logging.info(f"Timeline created successfully at {output_path}")
                    messagebox.showinfo("Success", f"{mode.title()} created at:\n{output_path}\n\nLog file: {log_file}")
            else:
                self.update_progress("Failed to create timeline", 0)
                logging.error("Failed to create timeline")
                messagebox.showerror("Error", "Failed to create timeline")
                
        except Exception as e:
            error_message = f"Error processing timeline: {str(e)}"
            logging.error(error_message, exc_info=True)  # This will include the full traceback
            self.update_progress(f"Error: {error_message}", 0)
            messagebox.showerror("Error", f"{error_message}\n\nCheck log file for details: {log_file}")

    def multi_approach_speech_detection(self, file_path):
        self.update_progress(f"Running multi-approach speech detection...", 30)
        logging.info("Running multi-approach speech detection...")
        all_segments = []
        
        # Determine file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Prepare log file for detailed debugging
        debug_log_path = os.path.join(output_path, f"multi_approach_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # Approach 1: Standard energy-based detection with multiple thresholds
        try:
            self.update_progress("Running energy-based detection with multiple thresholds...", 40)
            logging.info("Running energy-based detection with multiple thresholds...")
            base_threshold = self.sensitivity_var.get()
            
            # Try with multiple thresholds around the user-selected value
            thresholds = [base_threshold, base_threshold-10, base_threshold-20]
            for i, threshold_db in enumerate(thresholds):
                energy_segments = detect_audio_activity(
                    file_path, 
                    threshold_db=threshold_db,
                    chunk_length=20,  # Larger chunks
                    silence_margin=self.silence_margin_var.get(),
                    min_segment_duration=0.1  # 100ms minimum
                )
                all_segments.extend(energy_segments)
                logging.info(f"Energy detection (threshold {threshold_db}dB) found {len(energy_segments)} segments")
                self.update_progress(f"Energy detection (threshold {threshold_db}dB) found {len(energy_segments)} segments", 40 + i*5)
        except Exception as e:
            logging.error(f"Error in energy-based detection: {e}")
        
        # Approach 2: Voice isolation with band-pass filter
        try:
            self.update_progress("Running voice frequency isolation detection...", 55)
            logging.info("Running voice frequency isolation detection...")
            # Load the audio file
            if file_ext == '.mp3':
                audio = AudioSegment.from_mp3(file_path)
            elif file_ext == '.wav':
                audio = AudioSegment.from_wav(file_path)
            else:
                logging.error(f"Unsupported audio file type: {file_ext}")
                audio = None
            
            if audio:
                # Normalize audio if requested
                if self.normalize_audio_var.get():
                    logging.info("Normalizing audio volume...")
                    audio = effects.normalize(audio)
                
                # Convert to mono for simpler processing
                audio = audio.set_channels(1)
                
                # Apply band-pass filter to isolate voice frequencies
                logging.info("Applying band-pass filter (300Hz-3000Hz)...")
                filtered_audio = audio.high_pass_filter(300).low_pass_filter(3000)
                
                # Save the filtered audio temporarily
                temp_file = file_path.replace(file_ext, '_filtered.wav')
                filtered_audio.export(temp_file, format="wav")
                
                # Use regular detection on filtered audio
                isolation_segments = detect_audio_activity(
                    temp_file, 
                    threshold_db=self.sensitivity_var.get(),
                    silence_margin=self.silence_margin_var.get(),
                    min_segment_duration=0.1  # 100ms minimum
                )
                
                all_segments.extend(isolation_segments)
                logging.info(f"Voice isolation detection found {len(isolation_segments)} segments")
                self.update_progress(f"Voice isolation detection found {len(isolation_segments)} segments", 60)
                
                # Clean up temporary file
                os.remove(temp_file)
        except Exception as e:
            logging.error(f"Error in voice isolation detection: {e}")
        
        # Last resort approach for very quiet audio
        if not all_segments:
            self.update_progress("No segments detected. Trying more sensitive detection...", 70)
            logging.info("No segments detected. Trying more sensitive detection...")
            
            # Load audio file
            if file_ext == '.mp3':
                audio = AudioSegment.from_mp3(file_path)
            elif file_ext == '.wav':
                audio = AudioSegment.from_wav(file_path)
            else:
                logging.error(f"Unsupported audio file type: {file_ext}")
                audio = None
            
            if audio:
                # Normalize to maximize volume
                audio = effects.normalize(audio)
                
                # Use a very low threshold to detect any non-silence
                last_resort_segments = []
                chunk_ms = 500  # Larger chunks
                
                for i in range(0, len(audio), chunk_ms):
                    chunk = audio[i:i+chunk_ms]
                    # Check if there's any audio at all (very low threshold)
                    if chunk.rms > 10:  # Extremely low threshold
                        last_resort_segments.append((i/1000.0, (i+len(chunk))/1000.0))
                
                all_segments.extend(last_resort_segments)
                logging.info(f"Sensitive detection found {len(last_resort_segments)} segments")
                self.update_progress(f"Sensitive detection found {len(last_resort_segments)} segments", 75)
        
        # Add padding to all segments (50ms at beginning and end)
        padding = 0.05  # 50ms padding
        padded_segments = []
        for start, end in all_segments:
            padded_segments.append((max(0, start - padding), end + padding))
        
        # Merge all detected segments
        self.update_progress("Merging all detected segments with padding...", 80)
        logging.info("Merging all detected segments with padding...")
        # Use increased silence margin to ensure close segments are merged
        merged_segments = merge_speech_segments(padded_segments, threshold=self.silence_margin_var.get()*1.5)
        
        # Final filtering of very short segments
        log_file = os.path.join(output_path, f"final_segments_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        final_segments, removed_segments = debug_short_segments(
            merged_segments, 
            min_duration=0.1,  # 100 milliseconds minimum
            log_file=log_file
        )
        
        logging.info(f"Multi-approach detection found {len(final_segments)} final segments")
        if removed_segments:
            logging.info(f"Removed {len(removed_segments)} very short segments")
        
        # Log padding information
        logging.info(f"Applied {padding*1000}ms padding to beginning and end of all speech segments")
        
        return final_segments

    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Multi-Track Speech Processing")
        self.root.geometry("600x800")
        
        # Create a main canvas with scrollbar for the entire interface
        main_canvas = tk.Canvas(root)
        main_scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
        main_scrollable_frame = tk.Frame(main_canvas)
        
        # Configure the canvas and scrollable frame
        main_scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=main_scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=main_scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        main_scrollbar.pack(side="right", fill="y")
        
        # Get available audio tracks
        self.audio_tracks = get_audio_tracks()
        self.check_vars = []
        
        # Create frame for tracks
        tracks_frame = tk.Frame(main_scrollable_frame)
        tracks_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add label
        tk.Label(tracks_frame, text="Select Audio Tracks to Process:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Create scrollable area for tracks
        canvas = tk.Canvas(tracks_frame)
        scrollbar = tk.Scrollbar(tracks_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create checkboxes for audio tracks with track numbers and priority info
        for i, track in enumerate(self.audio_tracks):
            var = tk.BooleanVar()
            self.check_vars.append(var)
            priority_text = f" (Priority: {i+1})" if len(self.audio_tracks) > 1 else ""
            tk.Checkbutton(
                scrollable_frame, 
                text=f"Track {i+1}: {track}{priority_text}", 
                variable=var,
                font=("Arial", 10)
            ).pack(anchor='w', pady=2)
        
        # Options frame
        options_frame = tk.Frame(main_scrollable_frame)
        options_frame.pack(fill="x", expand=False, padx=10, pady=10)
        
        # Add detection methods group
        detection_frame = tk.LabelFrame(options_frame, text="Detection Method", font=("Arial", 11, "bold"), padx=10, pady=10)
        detection_frame.pack(fill="x", pady=5)
        
        self.detection_method_var = tk.StringVar(value="multi")
        tk.Radiobutton(detection_frame, text="Standard (Energy-based)", 
                    variable=self.detection_method_var, value="standard").pack(anchor='w')
        tk.Radiobutton(detection_frame, text="Voice Isolation", 
                    variable=self.detection_method_var, value="voice_isolation").pack(anchor='w')
        tk.Radiobutton(detection_frame, text="Multi-approach (Most Thorough)", 
                    variable=self.detection_method_var, value="multi").pack(anchor='w')
        
        # Add sensitivity controls
        sensitivity_frame = tk.LabelFrame(options_frame, text="Sensitivity Settings", font=("Arial", 11, "bold"), padx=10, pady=10)
        sensitivity_frame.pack(fill="x", pady=5)
        
        # Audio sensitivity slider
        tk.Label(sensitivity_frame, text="Audio Sensitivity:").pack(anchor='w')
        self.sensitivity_var = IntVar(value=-40)  # Default -40dB
        sensitivity_slider = Scale(
            sensitivity_frame, 
            from_=-60, 
            to=-20, 
            orient="horizontal",
            variable=self.sensitivity_var,
            length=350,
            tickinterval=10,
            resolution=5
        )
        sensitivity_slider.pack(fill="x", pady=(0, 10))
        
        # Silence margin slider
        tk.Label(sensitivity_frame, text="Silence Margin (sec):").pack(anchor='w')
        self.silence_margin_var = DoubleVar(value=0.8)  # Default 0.8 seconds
        margin_slider = Scale(
            sensitivity_frame,
            from_=0.1,
            to=2.0,
            orient="horizontal",
            variable=self.silence_margin_var,
            length=350,
            tickinterval=0.5,
            resolution=0.1
        )
        margin_slider.pack(fill="x")
        
        # Enhanced Interview Mode frame
        enhanced_frame = tk.LabelFrame(options_frame, text="Enhanced Interview Mode", font=("Arial", 11, "bold"), padx=10, pady=10)
        enhanced_frame.pack(fill="x", pady=5)
        
        # Enhanced mode checkbox
        self.enhanced_mode_var = tk.BooleanVar(value=True)
        self.enhanced_mode_check = tk.Checkbutton(
            enhanced_frame,
            text="Enable Enhanced Interview Mode (intelligent gap filling)",
            variable=self.enhanced_mode_var,
            font=("Arial", 10),
            command=self.toggle_gap_filling_controls
        )
        self.enhanced_mode_check.pack(anchor='w', pady=5)
        
        # Gap filling controls (shown when enhanced mode is enabled)
        self.gap_frame = tk.Frame(enhanced_frame)
        tk.Label(self.gap_frame, text="Minimum Gap Duration to Fill (sec):").pack(anchor='w')
        self.min_gap_var = DoubleVar(value=0.8)  # Default 0.8 seconds
        gap_slider = Scale(
            self.gap_frame,
            from_=0.3,
            to=2.0,
            orient="horizontal",
            variable=self.min_gap_var,
            length=350,
            tickinterval=0.5,
            resolution=0.1
        )
        gap_slider.pack(fill="x")
        
        # Help text for enhanced mode
        help_text = tk.Label(self.gap_frame, 
                           text="Enhanced mode fills silent gaps with speech from other tracks.\n"
                                "Higher track numbers have priority (Track 3 > Track 2 > Track 1).",
                           font=("Arial", 9),
                           fg="gray",
                           justify="left")
        help_text.pack(anchor='w', pady=(5, 0))
        
        # Advanced options frame
        advanced_frame = tk.LabelFrame(options_frame, text="Advanced Options", font=("Arial", 11, "bold"), padx=10, pady=10)
        advanced_frame.pack(fill="x", pady=5)
        
        # Voice isolation frequency range (only shown when voice isolation is selected)
        self.low_freq_var = IntVar(value=300)
        self.high_freq_var = IntVar(value=3000)
        
        self.freq_frame = tk.Frame(advanced_frame)
        tk.Label(self.freq_frame, text="Voice Frequency Range:").pack(anchor='w')
        freq_controls = tk.Frame(self.freq_frame)
        freq_controls.pack(fill="x")
        
        tk.Label(freq_controls, text="Low:").pack(side="left")
        tk.Entry(freq_controls, textvariable=self.low_freq_var, width=5).pack(side="left", padx=(0, 10))
        tk.Label(freq_controls, text="High:").pack(side="left")
        tk.Entry(freq_controls, textvariable=self.high_freq_var, width=5).pack(side="left")
        tk.Label(freq_controls, text="Hz").pack(side="left")
        
        # Additional options - default normalize to enabled
        self.normalize_audio_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame,
            text="Normalize audio before detection (helps with quiet audio)",
            variable=self.normalize_audio_var,
            font=("Arial", 10)
        ).pack(anchor='w', pady=5)
        
        # Add single track mode option
        self.single_track_mode_var = tk.BooleanVar(value=False)
        self.single_track_mode_check = tk.Checkbutton(
            advanced_frame,
            text="Single Track Mode (create highlights from one track)",
            variable=self.single_track_mode_var,
            font=("Arial", 10)
        )
        self.single_track_mode_check.pack(anchor='w', pady=5)
        
        # Add a separator
        tk.Frame(advanced_frame, height=1, bg="gray").pack(fill="x", pady=5)
        
        # Add timeline splitting option - default enabled
        self.split_timeline_var = tk.BooleanVar(value=True)
        self.split_checkbox = tk.Checkbutton(
            advanced_frame,
            text="Split timeline into chunks",
            variable=self.split_timeline_var,
            font=("Arial", 10),
            command=self.toggle_chunk_controls
        )
        self.split_checkbox.pack(anchor='w', pady=5)
        
        # Add chunk duration controls (initially hidden)
        self.chunk_frame = tk.Frame(advanced_frame)
        tk.Label(self.chunk_frame, text="Target chunk duration (minutes):").pack(anchor='w')
        
        self.chunk_duration_var = IntVar(value=10)  # Default 10 minutes
        chunk_slider = Scale(
            self.chunk_frame,
            from_=5,
            to=30,
            orient="horizontal",
            variable=self.chunk_duration_var,
            length=350,
            tickinterval=5,
            resolution=5
        )
        chunk_slider.pack(fill="x")
        
        # Add progress frame with label and progress bar
        progress_frame = tk.Frame(main_scrollable_frame)
        progress_frame.pack(fill="x", expand=False, padx=10, pady=10)
        
        self.progress_label = tk.Label(progress_frame, text="Ready to process", font=("Arial", 10))
        self.progress_label.pack(pady=(5, 5), anchor='w')
        
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var,
            orient="horizontal",
            length=500,
            mode="determinate"
        )
        self.progress_bar.pack(fill="x", pady=(0, 5))
        
        # Buttons frame
        buttons_frame = tk.Frame(main_scrollable_frame)
        buttons_frame.pack(fill="x", expand=False, padx=10, pady=10)
        
        # Process button
        process_button = tk.Button(
            buttons_frame, 
            text="Process Selected Tracks", 
            command=self.process_selected_tracks,
            font=("Arial", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10
        )
        process_button.pack(side="left", padx=(0, 10))
        
        # Help button
        help_button = tk.Button(
            buttons_frame,
            text="Help",
            command=self.show_help,
            font=("Arial", 10),
            padx=10,
            pady=5)
        help_button.pack(side="left")
        
        # Only show frequency controls when voice isolation is selected
        self.detection_method_var.trace_variable("w", self.toggle_freq_controls)
        
        # Initialize control visibility
        self.toggle_freq_controls()
        self.toggle_chunk_controls()
        self.toggle_gap_filling_controls()

# Legacy function for backwards compatibility
def modify_timeline_with_interview_style(otio_path, track_segments_dict):
    """Legacy function that calls the enhanced version with default timeline duration"""
    timeline_duration = get_timeline_duration(otio_path)
    return modify_timeline_with_enhanced_interview_style(otio_path, track_segments_dict, timeline_duration)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRenderGUI(root)
    root.mainloop()