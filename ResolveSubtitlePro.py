import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pysrt
import logging
import datetime
import spacy
import tempfile
import shutil

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='subtitle_generator_debug.log',
    filemode='w'
)
logger = logging.getLogger('SubtitleGenerator')

# Add console handler to see logs in real-time
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)


def load_spacy_model():
    """Load the spaCy model, installing it if necessary."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except IOError:
        # Try to install the model
        import subprocess
        try:
            subprocess.check_call([
                "python", "-m", "spacy", "download", "en_core_web_sm"
            ])
            # Try loading again
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except Exception as e:
            raise

class SubtitleSegmenter:
    def __init__(self):
        """Initialize the SubtitleSegmenter with the spaCy model."""
        self.nlp = load_spacy_model()
        
        # Define common contractions to protect
        self.contractions = [
            "that's", "it's", "he's", "she's", "we're", "you're", "they're",
            "i'm", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't",
            "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
            "haven't", "hasn't", "hadn't", "i've", "you've", "we've", "they've",
            "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "i'd",
            "you'd", "he'd", "she'd", "we'd", "they'd", "there's", "here's",
            "what's", "where's", "when's", "who's", "how's", "why's", "let's"
        ]
        
        # Define preferred break patterns - ordered by priority
        self.preferred_breaks = [
            # 1. Breaking at natural clause boundaries
            {"pos": ["CCONJ"], "text": ["and", "but", "or"], "dep": ["cc"], "prefix": True},
            
            # 2. Breaking after a complete subject-verb-object
            {"pos_seq": [["NOUN", "PROPN", "PRON"], ["VERB"], ["NOUN", "PROPN", "PRON"]]},
            
            # 3. Breaking after a verb before a prepositional phrase
            {"pos_seq": [["VERB"], ["ADP"]]},
            
            # 4. Breaking after certain adverbials
            {"pos": ["ADV"], "text": ["then", "therefore", "however", "meanwhile"], "prefix": True},
            
            # 5. Breaking before prepositions that start new phrases
            {"pos": ["ADP"], "text": ["with", "without", "by", "in", "on", "at"], "prefix": True}
        ]
        
        # Define linguistic patterns that should NEVER be broken
        self.no_break_patterns = [
            # 1. Don't break between determiner and noun
            {"pos_seq": [["DET"], ["NOUN", "PROPN"]]},
            
            # 2. Don't break between adjective and noun
            {"pos_seq": [["ADJ"], ["NOUN", "PROPN"]]},
            
            # 3. Don't break in phrasal verbs
            {"pos_seq": [["VERB"], ["ADP", "ADV"]], "dep": ["prt", "compound"]},
            
            # 4. Don't break between auxiliary and main verb
            {"pos_seq": [["AUX"], ["VERB"]]},
            
            # 5. Don't break between verb and direct object
            {"pos_seq": [["VERB"], ["DET", "PRON", "ADJ", "NOUN", "PROPN"]]},
            
            # 6. WH-Questions: Don't break between question words and their clauses
            {"pos_seq": [["ADV"], ["PRON"]], "text": ["how", "what", "where", "when", "why"]},
            {"pos_seq": [["ADV"], ["AUX"]], "text": ["how", "what", "where", "when", "why"]},
            {"pos_seq": [["ADV"], ["VERB"]], "text": ["how", "what", "where", "when", "why"]},
            {"pos_seq": [["ADV"], ["DET"]], "text": ["how", "what", "where", "when", "why"]},
            {"pos_seq": [["ADV"], ["ADJ"]], "text": ["how", "what", "where", "when", "why"]},
            
            # 7. Question pronouns: who, what, which + auxiliary/verb
            {"pos_seq": [["PRON"], ["AUX"]], "text": ["who", "what", "which"]},
            {"pos_seq": [["PRON"], ["VERB"]], "text": ["who", "what", "which"]},
            
            # 8. Don't break between pronoun and auxiliary in questions
            {"pos_seq": [["PRON"], ["AUX"]], "text": ["it", "that", "this", "he", "she", "we", "they", "you", "i"]},
            
            # 9. Auxiliary verb patterns in yes/no questions
            {"pos_seq": [["AUX"], ["PRON"]], "text": ["do", "does", "did", "is", "are", "was", "were", "have", "has", "had", "will", "would", "can", "could", "should", "might", "may"]},
            
            # 10. Modal verbs with subjects
            {"pos_seq": [["VERB"], ["PRON"]], "text": ["can", "could", "will", "would", "should", "might", "may", "must"]},
            
            # 11. Question tags - don't separate tag from main clause
            {"pos_seq": [["PRON"], ["AUX"]], "text": ["you", "he", "she", "it", "we", "they"]},
            
            # 12. Compound question words
            {"text": ["how much", "how many", "how long", "how far", "how often", "how old", "what time", "what kind", "which one"]},
            
            # 13. Embedded question patterns - keep polite question starters together
            {"text": ["could you tell me", "do you know", "can you tell me", "would you mind", "i wonder"]},
            
            # 14. Negative questions - keep negative with auxiliary
            {"pos_seq": [["PART"], ["AUX"]], "text": ["not", "n't"]},
            {"text": ["don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't", "mightn't"]},
        ]

    def _protect_contractions(self, text):
        """
        Protect contractions from being split by replacing them with placeholders.
        Returns: (protected_text, contraction_map)
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        protected_text = text
        contraction_map = {}
        
        for i, contraction in enumerate(self.contractions):
            placeholder = f"<CONTRACTION_{i}>"
            
            # Case-insensitive replacement but preserve original case
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            
            def replace_func(match):
                original = match.group(0)
                contraction_map[placeholder] = original
                logger.debug(f"Protected contraction: '{original}' -> '{placeholder}'")
                return placeholder
            
            protected_text = pattern.sub(replace_func, protected_text)
        
        return protected_text, contraction_map

    def _restore_contractions(self, text, contraction_map):
        """
        Restore protected contractions from placeholders.
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        restored_text = text
        for placeholder, original in contraction_map.items():
            restored_text = restored_text.replace(placeholder, original)
            logger.debug(f"Restored contraction: '{placeholder}' -> '{original}'")
        
        return restored_text

    def identify_break_points(self, doc, target_length=(4, 6)):
        """
        Identify break points for linguistic segmentation.
        This method ONLY breaks at sentence boundaries (periods, exclamation marks, question marks).
        Enhanced to respect contraction boundaries.
        """
        min_length, max_length = target_length
        mandatory_breaks = []

        # ONLY PASS: Identify sentence boundaries as break points
        for i in range(len(doc)):
            # Check if this token is sentence-ending punctuation
            if doc[i].text in ['.', '!', '?'] and not doc[i].is_space:
                
                # CRITICAL: Don't break if this punctuation is part of a contraction placeholder
                if i > 0 and "<CONTRACTION_" in doc[i-1].text:
                    logger.debug(f"Skipping break after punctuation in contraction placeholder at position {i}")
                    continue
                
                # Make sure this isn't an abbreviation by checking if next token exists and context
                if i < len(doc) - 1:  # Not the last token
                    next_token = doc[i + 1]
                    # This is a sentence boundary if next token starts a new sentence or has whitespace
                    if (next_token.text and next_token.text[0].isupper()) or next_token.whitespace_:
                        
                        # Additional check: make sure we're not breaking within a contraction
                        break_position = i + 1
                        if break_position < len(doc):
                            # Check if the break would split a contraction placeholder
                            context_before = ''.join([doc[j].text for j in range(max(0, break_position - 2), break_position)])
                            context_after = ''.join([doc[j].text for j in range(break_position, min(len(doc), break_position + 2))])
                            
                            if "<CONTRACTION_" in context_before or "<CONTRACTION_" in context_after:
                                logger.debug(f"Skipping break at position {break_position} - would split contraction")
                                continue
                        
                        mandatory_breaks.append(break_position)  # Break AFTER the punctuation
                else:
                    # This is the last token and it's punctuation, so it ends the text
                    # No break needed as this is the natural end
                    pass

        # Remove duplicates and sort
        mandatory_breaks = sorted(list(set(mandatory_breaks)))
        
        # Filter out breaks that would create segments that are too short
        filtered_breaks = []
        segment_boundaries = [0] + mandatory_breaks + [len(doc)]
        
        for i in range(len(segment_boundaries) - 1):
            seg_start = segment_boundaries[i]
            seg_end = segment_boundaries[i + 1]
            
            # Count actual words (not punctuation) in this segment
            word_count = sum(1 for j in range(seg_start, seg_end) 
                            if not doc[j].is_punct and not doc[j].is_space and not doc[j].text.startswith('<CONTRACTION_'))

            # If this segment has at least 2 words, keep the break that creates it
            if word_count >= 2 and i < len(mandatory_breaks):
                filtered_breaks.append(mandatory_breaks[i])
        
        # Log the decision
        logger.debug(f"Linguistic segmentation found {len(filtered_breaks)} sentence boundaries")
        for break_pos in filtered_breaks:
            if break_pos < len(doc):
                context_start = max(0, break_pos - 5)
                context_end = min(len(doc), break_pos + 5)
                context = ''.join([doc[j].text for j in range(context_start, break_pos)]) + '|' + ''.join([doc[j].text for j in range(break_pos, context_end)])
                logger.debug(f"  Break at position {break_pos}: ...{context}...")
        
        return filtered_breaks

    def _is_valid_break(self, doc, position):
        """
        Check if a position is a linguistically valid break point.
        """
        # NEVER break at punctuation marks
        if doc[position].is_punct:
            return False
            
        # NEVER break after a determiner (highest priority rule)
        if position > 0 and doc[position-1].pos_ == "DET":
            return False
            
        # Check against no-break patterns
        for pattern in self.no_break_patterns:
            if "pos_seq" in pattern:
                # Check if this position matches a no-break sequence
                prev_token = doc[position-1]
                curr_token = doc[position]
                
                for allowed_prev in pattern["pos_seq"][0]:
                    for allowed_curr in pattern["pos_seq"][1]:
                        if prev_token.pos_ == allowed_prev and curr_token.pos_ == allowed_curr:
                            # Check if there's a dependency constraint
                            if "dep" not in pattern or curr_token.dep_ in pattern["dep"]:
                                logger.debug(f"Invalid break at {position}: {prev_token.text} ({prev_token.pos_}) -> {curr_token.text} ({curr_token.pos_})")
                                return False
        
        return True

    def _calculate_break_score(self, doc, position):
        """
        Calculate a score for a break point based on linguistic features.
        Higher scores are better break points.
        """
        score = 0
        curr_token = doc[position]
        prev_token = doc[position-1]
        
        # CRITICAL FIX: Never break between pronoun and verb
        # This prevents splits like "what I | just said" 
        if (prev_token.pos_ in ["PRON", "PROPN"] and 
            curr_token.pos_ in ["VERB", "AUX"] and 
            curr_token.dep_ in ["ROOT", "aux", "auxpass"]):
            score -= 1000  # Massive penalty to make this extremely unlikely
        
        # CRITICAL FIX: Never break between subject pronoun and predicate
        if (prev_token.pos_ == "PRON" and prev_token.dep_ in ["nsubj", "nsubjpass"] and
            position < len(doc) and curr_token.dep_ in ["ROOT", "aux", "auxpass", "advcl"]):
            score -= 1000
        
        # BONUS: Prefer breaking after complete clauses
        if prev_token.dep_ == "ROOT" and curr_token.pos_ in ["PRON", "PROPN", "NOUN"]:
            logger.debug(f"Bonus: breaking after complete clause ending with '{prev_token.text}'")
            score += 100
        
        # BONUS: Prefer breaking after objects
        if prev_token.dep_ in ["dobj", "pobj", "iobj"] and curr_token.pos_ in ["PRON", "PROPN"]:
            logger.debug(f"Bonus: breaking after object '{prev_token.text}' before '{curr_token.text}'")
            score += 50
        
        # Check preferred breaks by pattern matching (original logic)
        for i, pattern in enumerate(self.preferred_breaks):
            # Higher priority patterns get higher base scores
            base_score = 50 - (i * 10)
            
            matched = False
            
            # Check token to break before (prefix=True) or after (prefix=False)
            check_token = curr_token if pattern.get("prefix", False) else prev_token
            
            # Check part of speech
            if "pos" in pattern and check_token.pos_ in pattern["pos"]:
                matched = True
                
            # Check specific text
            if "text" in pattern and check_token.text.lower() in pattern["text"]:
                matched = True
                score += 5  # Bonus for matching specific text
                
            # Check dependency
            if "dep" in pattern and check_token.dep_ in pattern["dep"]:
                matched = True
                
            # Check POS sequences
            if "pos_seq" in pattern:
                seq_match = True
                for j, allowed_pos in enumerate(pattern["pos_seq"]):
                    token_pos = position - 1 + j
                    if token_pos >= len(doc) or doc[token_pos].pos_ not in allowed_pos:
                        seq_match = False
                        break
                if seq_match:
                    matched = True
                    score += 10  # Bonus for matching sequences
            
            if matched:
                score += base_score
        
        # Adjust score based on ideal positioning
        # Prefer breaks that result in more balanced segments
        return score

    def segment_text(self, text, target_length=(4, 6)):
        """
        Enhanced segment_text with contraction protection.
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        # Clean the text and ensure proper spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Check if text is too short to split
        words = text.split()
        if len(words) <= target_length[1]:
            return [text]
        
        logger.debug(f"Segmenting text with contraction protection: '{text}'")
        
        # STEP 1: Protect contractions
        protected_text, contraction_map = self._protect_contractions(text)
        logger.debug(f"Protected text: '{protected_text}'")
        
        # STEP 2: Process with spaCy using protected text
        doc = self.nlp(protected_text)
        
        # STEP 3: Get optimal break points (now respecting contractions)
        break_points = self.identify_break_points(doc, target_length)
        
        # STEP 4: Create segments based on break points
        segments = []
        start_idx = 0
        
        for break_idx in sorted(break_points):
            # Create segment text more carefully to handle punctuation
            segment_tokens = doc[start_idx:break_idx+1]
            
            # Extract the raw text and clean it up
            segment_text = segment_tokens.text.strip()
            
            # CRITICAL FIX: Handle comma placement correctly
            if break_idx + 1 < len(doc):
                next_token = doc[break_idx + 1]
                if next_token.text.strip() == ',':
                    segment_text += ','
                    start_idx = break_idx + 2
                else:
                    start_idx = break_idx + 1
            else:
                start_idx = break_idx + 1
            
            if segment_text:
                segments.append(segment_text)
        
        # Add the final segment if there are remaining tokens
        if start_idx < len(doc):
            final_tokens = doc[start_idx:]
            final_text = final_tokens.text.strip()
            
            # Clean up any leading punctuation from the final segment
            final_text = re.sub(r'^[,\s]+', '', final_text)
            
            if final_text:
                segments.append(final_text)
        
        # STEP 5: Restore contractions in all segments
        restored_segments = []
        for segment in segments:
            restored_segment = self._restore_contractions(segment, contraction_map)
            restored_segments.append(restored_segment)
        
        # POST-PROCESSING: Fix poor splits by rejoining segments (now with restored contractions)
        restored_segments = self._fix_poor_splits(restored_segments)
        
        # FINAL CLEANUP: Ensure no segment starts with a comma
        cleaned_segments = []
        for segment in restored_segments:
            cleaned_segment = re.sub(r'^[,\s]+', '', segment).strip()
            if cleaned_segment:
                cleaned_segments.append(cleaned_segment)
        
        logger.debug(f"Final segments with restored contractions: {cleaned_segments}")
        
        return cleaned_segments if cleaned_segments else [text]

    def _fix_poor_splits(self, segments):
        """
        Post-process segments to fix poor splits like orphaned conjunctions.
        This keeps phrases like 'or something', 'and then', 'but wait' together.
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        if len(segments) <= 1:
            return segments
        
        # Define patterns that should stay together
        keep_together_patterns = [
            # Conjunction + content patterns
            {'first': ['or'], 'second_starts': ['something', 'anything', 'nothing', 'someone', 'anyone']},
            {'first': ['and'], 'second_starts': ['then', 'now', 'so', 'also', 'maybe', 'perhaps']},
            {'first': ['but'], 'second_starts': ['wait', 'still', 'then', 'now', 'also', 'maybe']},
            {'first': ['so'], 'second_starts': ['then', 'now', 'what', 'maybe', 'perhaps']},
            
            # Short connectors that shouldn't be alone
            {'orphaned_words': ['or', 'and', 'but', 'so', 'yet', 'for', 'nor']},
            
            # Common phrase beginnings
            {'first': ['you'], 'second_starts': ['know', 'see', 'think', 'mean']},
            {'first': ['i'], 'second_starts': ['mean', 'think', 'guess', 'know']},
            {'first': ['let'], 'second_starts': ['me', 'us', 'them']},
            {'first': ['kind'], 'second_starts': ['of']},
            {'first': ['sort'], 'second_starts': ['of']},
        ]
        
        fixed_segments = []
        i = 0
        
        while i < len(segments):
            current_segment = segments[i].strip()
            next_segment = segments[i + 1].strip() if i + 1 < len(segments) else None
            
            logger.debug(f"Analyzing segment {i+1}: '{current_segment}' + next: '{next_segment}'")
            
            should_merge = False
            merge_reason = ""
            
            if next_segment:
                current_words = current_segment.split()
                next_words = next_segment.split()
                
                # Check each pattern
                for pattern in keep_together_patterns:
                    # Check for orphaned words that shouldn't be alone
                    if 'orphaned_words' in pattern:
                        if (len(current_words) == 1 and 
                            current_words[0].lower().strip('.,!?;:') in pattern['orphaned_words']):
                            should_merge = True
                            merge_reason = f"orphaned word '{current_words[0]}'"
                            break
                    
                    # Check for first+second patterns
                    if 'first' in pattern and 'second_starts' in pattern:
                        if (len(current_words) >= 1 and next_words and
                            current_words[-1].lower().strip('.,!?;:') in pattern['first'] and
                            next_words[0].lower().strip('.,!?;:') in pattern['second_starts']):
                            should_merge = True
                            merge_reason = f"pattern '{current_words[-1]} {next_words[0]}'"
                            break
            
            if should_merge and next_segment:
                # Merge current and next segments
                merged_segment = f"{current_segment} {next_segment}"
                logger.info(f"Merging segments to fix poor split ({merge_reason}):")
                logger.info(f"  '{current_segment}' + '{next_segment}' → '{merged_segment}'")
                fixed_segments.append(merged_segment)
                i += 2  # Skip the next segment since we merged it
            else:
                # Keep current segment as is
                fixed_segments.append(current_segment)
                i += 1
        
        return fixed_segments

    def process_srt_file(self, input_file, output_file, target_length=(4, 6)):
        """
        Process an SRT file, segmenting long subtitles into shorter ones while preserving original timing.
        Enhanced with contraction protection.
        
        Args:
            input_file (str): Path to input SRT file
            output_file (str): Path to output SRT file
            target_length (tuple): Target min and max words per segment
            
        Returns:
            bool: Success or failure
        """
        try:
            logger.info(f"Processing SRT file: {input_file} with target length {target_length}")
            subs = pysrt.open(input_file, encoding='utf-8')
            new_subs = pysrt.SubRipFile()
            counter = 1
            
            for sub_index, sub in enumerate(subs):
                text = sub.text.replace("\n", " ").strip()
                
                # Skip empty subtitles
                if not text:
                    logger.debug(f"Skipping empty subtitle {sub_index+1}")
                    continue

                # Count words
                word_count = len(text.split())
                
                # If subtitle is already short enough, keep it as is
                if word_count <= target_length[1]:
                    sub.index = counter
                    new_subs.append(sub)
                    counter += 1
                    continue
                
                # Segment the subtitle (now with contraction protection)
                segments = self.segment_text(text, target_length)
                
                # CRITICAL FIX: Preserve original timing properly
                original_start_ms = sub.start.ordinal
                original_end_ms = sub.end.ordinal
                original_duration = original_end_ms - original_start_ms

                # Calculate proportional timing based on character count
                total_chars = sum(len(segment) for segment in segments)
                
                # Create new subtitles for each segment with proper timing
                current_start_ms = original_start_ms
                
                for i, segment in enumerate(segments):
                    segment_chars = len(segment)
                    
                    # Calculate proportional duration
                    if i == len(segments) - 1:
                        # Last segment gets remaining time
                        segment_end_ms = original_end_ms
                    else:
                        # Proportional duration based on character count
                        proportion = segment_chars / total_chars
                        segment_duration = int(proportion * original_duration)
                        
                        # Ensure minimum duration of 500ms
                        segment_duration = max(500, segment_duration)
                        
                        segment_end_ms = current_start_ms + segment_duration
                    
                    # Ensure we don't exceed original end time
                    segment_end_ms = min(segment_end_ms, original_end_ms)
                    
                    # Ensure start < end
                    if current_start_ms >= segment_end_ms:
                        segment_end_ms = current_start_ms + 500  # Minimum 500ms duration

                    # Create new subtitle
                    new_sub = pysrt.SubRipItem(
                        index=counter,
                        start=pysrt.SubRipTime.from_ordinal(current_start_ms),
                        end=pysrt.SubRipTime.from_ordinal(segment_end_ms),
                        text=segment
                    )
                    new_subs.append(new_sub)
                    counter += 1
                    
                    # Next segment starts where this one ends (or shortly after for small gap)
                    current_start_ms = segment_end_ms
            
            # Save the new subtitles
            new_subs.save(output_file, encoding='utf-8')
            return True
            
        except Exception as e:
            logger.error(f"Error processing SRT file: {e}", exc_info=True)
            return False

class SubtitleGenerator:
    def setup_ui(self):
        """Create the UI elements"""
        logger = logging.getLogger('SubtitleGenerator')
        logger.debug("Setting up UI elements")
        
        # Title label
        title_label = tk.Label(self.root, text="Simple Subtitle Generator", font=("Arial", 16))
        title_label.pack(pady=10)
        
        # SRT file selection - MODIFIED for multiple files with templates
        srt_selection_frame = tk.LabelFrame(self.root, text="SRT Files & Templates")
        srt_selection_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # Buttons frame
        buttons_frame = tk.Frame(srt_selection_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.select_multiple_button = tk.Button(buttons_frame, text="Add SRT Files", command=self.select_multiple_srts)
        self.select_multiple_button.pack(side=tk.LEFT, padx=5)
        
        self.remove_selected_button = tk.Button(buttons_frame, text="Remove Selected", command=self.remove_selected_files)
        self.remove_selected_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_files_button = tk.Button(buttons_frame, text="Clear All", command=self.clear_srt_files)
        self.clear_files_button.pack(side=tk.LEFT, padx=5)
        
        # Files and templates frame with scrollbar
        list_frame = tk.Frame(srt_selection_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create main frame for the treeview
        tree_frame = tk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Treeview for files and templates
        columns = ('File', 'Template')
        self.files_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=8)
        
        # Define headings
        self.files_tree.heading('File', text='SRT File')
        self.files_tree.heading('Template', text='Text+ Template')
        
        # Configure column widths
        self.files_tree.column('File', width=250, minwidth=150)
        self.files_tree.column('Template', width=200, minwidth=100)
        
        # Add scrollbar for treeview
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        self.files_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Pack treeview and scrollbar
        self.files_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to change template
        self.files_tree.bind('<Double-1>', self.on_tree_double_click)
        
        # Template selection instructions
        instructions_label = tk.Label(srt_selection_frame, 
                                    text="Double-click on a template cell to change it", 
                                    font=("Arial", 9), fg="gray")
        instructions_label.pack(pady=2)
        
        # Status label for file count
        self.file_count_label = tk.Label(srt_selection_frame, text="No SRT files selected", font=("Arial", 9), fg="gray")
        self.file_count_label.pack(pady=2)
        
        # Preprocessing options
        preprocess_frame = tk.LabelFrame(self.root, text="Preprocessing Options")
        preprocess_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Smart comma replacement option
        self.smart_comma_mode_var = tk.BooleanVar(value=True)
        self.smart_comma_check = tk.Checkbutton(
            preprocess_frame, 
            text="Use smart comma-to-period replacement (considers interjections)", 
            variable=self.smart_comma_mode_var
        )
        self.smart_comma_check.pack(anchor=tk.W, padx=5, pady=2)
        
        # Split long subtitles option
        self.split_long_var = tk.BooleanVar(value=True)
        self.split_long_check = tk.Checkbutton(preprocess_frame, text="Split long subtitles", 
                                            variable=self.split_long_var)
        self.split_long_check.pack(anchor=tk.W, padx=5, pady=2)
        
        # Split at punctuation option
        self.split_punct_var = tk.BooleanVar(value=True)
        self.split_punct_check = tk.Checkbutton(preprocess_frame, text="Split at punctuation", 
                                            variable=self.split_punct_var)
        self.split_punct_check.pack(anchor=tk.W, padx=5, pady=2)
        
        # Split at commas option
        self.split_commas_var = tk.BooleanVar(value=True)
        self.split_commas_check = tk.Checkbutton(preprocess_frame, text="Split at commas", 
                                                variable=self.split_commas_var)
        self.split_commas_check.pack(anchor=tk.W, padx=5, pady=2)
        
        # Add commas before interjections option
        self.add_commas_var = tk.BooleanVar(value=True)
        self.add_commas_check = tk.Checkbutton(preprocess_frame, text="Add commas before interjections", 
                                            variable=self.add_commas_var)
        self.add_commas_check.pack(anchor=tk.W, padx=5, pady=2)
        
        # Add commas after starting interjections option
        self.add_commas_after_var = tk.BooleanVar(value=True)
        self.add_commas_after_check = tk.Checkbutton(preprocess_frame, text="Add commas after starting interjections", 
                                                variable=self.add_commas_after_var)
        self.add_commas_after_check.pack(anchor=tk.W, padx=5, pady=2)
        
        # Add linguistic segmentation option
        self.linguistic_segment_var = tk.BooleanVar(value=True)
        self.linguistic_segment_check = tk.Checkbutton(preprocess_frame, text="Use linguistic segmentation (spaCy)", 
                                                    variable=self.linguistic_segment_var)
        self.linguistic_segment_check.pack(anchor=tk.W, padx=5, pady=2)
        
        # Target length frame for linguistic segmentation
        target_frame = tk.Frame(preprocess_frame)
        target_frame.pack(fill=tk.X, padx=20, pady=2)
        
        target_label = tk.Label(target_frame, text="Target words per segment (min-max):")
        target_label.pack(side=tk.LEFT, padx=5)
        
        self.min_words_var = tk.StringVar(value="4")
        self.min_words_entry = tk.Entry(target_frame, textvariable=self.min_words_var, width=3)
        self.min_words_entry.pack(side=tk.LEFT, padx=2)
        
        dash_label = tk.Label(target_frame, text="-")
        dash_label.pack(side=tk.LEFT)
        
        self.max_words_var = tk.StringVar(value="6")
        self.max_words_entry = tk.Entry(target_frame, textvariable=self.max_words_var, width=3)
        self.max_words_entry.pack(side=tk.LEFT, padx=2)
        
        # Add counters for consecutive duplicates option
        self.add_duplicate_counters_var = tk.BooleanVar(value=True)
        self.add_duplicate_counters_check = tk.Checkbutton(preprocess_frame, text="Add counters for consecutive duplicate words (X2, X3, etc.)", 
                                                        variable=self.add_duplicate_counters_var)
        self.add_duplicate_counters_check.pack(anchor=tk.W, padx=5, pady=2)
        
        # Subtitle delay option
        delay_frame = tk.LabelFrame(self.root, text="Timing Options")
        delay_frame.pack(fill=tk.X, padx=20, pady=5)
        
        delay_option_frame = tk.Frame(delay_frame)
        delay_option_frame.pack(fill=tk.X, padx=5, pady=2)
        
        delay_label = tk.Label(delay_option_frame, text="Subtitle delay (seconds):")
        delay_label.pack(side=tk.LEFT, padx=5)
        
        self.delay_var = tk.StringVar(value="0.2")
        self.delay_entry = tk.Entry(delay_option_frame, textvariable=self.delay_var, width=5)
        self.delay_entry.pack(side=tk.LEFT, padx=5)
        
        delay_help_label = tk.Label(delay_option_frame, text="(0 = no delay, 0.5 = half second delay)", 
                                font=("Arial", 8), fg="gray")
        delay_help_label.pack(side=tk.LEFT, padx=5)
        
        # Processing options frame
        processing_frame = tk.LabelFrame(self.root, text="Processing Options")
        processing_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Radio buttons for processing mode
        self.processing_mode_var = tk.StringVar(value="sequential")
        
        sequential_radio = tk.Radiobutton(processing_frame, text="Process files sequentially (each file on separate tracks)", 
                                        variable=self.processing_mode_var, value="sequential")
        sequential_radio.pack(anchor=tk.W, padx=5, pady=2)
        
        batch_radio = tk.Radiobutton(processing_frame, text="Process all files as one combined batch", 
                                variable=self.processing_mode_var, value="batch")
        batch_radio.pack(anchor=tk.W, padx=5, pady=2)
        
        # Add subtitles button
        self.add_subs_button = tk.Button(self.root, text="Add Subtitles to Timeline", command=self.process_subtitles)
        self.add_subs_button.config(state="disabled")  # Disabled until SRT file is selected
        self.add_subs_button.pack(pady=10)
        
        # Progress bar for multiple file processing
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=5)
        self.progress_bar.pack_forget()  # Hide initially
        
        # Status message
        self.status_label = tk.Label(self.root, text="Ready", font=("Arial", 10))
        self.status_label.pack(pady=15)
        
        logger.debug("UI setup complete")

    def __init__(self, root):
        logger.info("Initializing SubtitleGenerator")
        self.root = root
        self.root.title("Simple Subtitle Generator - Multiple Files")
        self.root.geometry("600x900")  # Made window larger to accommodate test button

        # Store file paths and templates - MODIFIED for multiple files with individual templates
        self.file_template_pairs = []  # List of (srt_path, template_object) tuples
        self.processed_srt_paths = []  # List of processed SRT paths
        
        # Find Text+ templates in media pool FIRST
        self.mediaPoolItemsList = []
        logger.debug("Searching for Text+ templates")
        self.find_text_templates()
        
        # Configure window AFTER finding templates
        logger.debug("Setting up UI")
        self.setup_ui()
        
        # Initialize subtitle segmenter (only when needed)
        self.subtitle_segmenter = None
        logger.info("Initialization complete")

    def add_subtitles(self):
        """Add subtitles to timeline - core functionality from original script"""
        logger.info("Starting add_subtitles process")
        process_start_time = datetime.datetime.now()
        
        try:
            self.status_label.config(text="Adding subtitles, please wait...")
            self.root.update()  # Update UI
            
            # Get subtitle delay from UI
            try:
                subtitle_delay_seconds = float(self.delay_var.get())
                logger.info(f"Using subtitle delay: {subtitle_delay_seconds} seconds")
            except ValueError:
                logger.warning("Invalid delay value, using default 0.2 seconds")
                subtitle_delay_seconds = 0.2
            
            # Preprocess the SRT file if any preprocessing options are selected
            if self.split_long_var.get() or self.split_punct_var.get() or self.split_commas_var.get() or self.add_commas_var.get() or self.add_commas_after_var.get():
                logger.info("Preprocessing options selected, calling preprocess_srt()")
                if not self.preprocess_srt():
                    logger.error("Preprocessing failed, aborting")
                    return
            else:
                logger.debug("No preprocessing options selected, using original SRT file")
                self.processed_srt_path = self.srt_file_path
            
            # Verify project and timeline
            logger.debug("Verifying project and timeline")
            project_manager = resolve.GetProjectManager()
            project = project_manager.GetCurrentProject()
            if not project:
                logger.error("No project found")
                self.show_error("No project found")
                return
            logger.debug("Project found")
            
            media_pool = project.GetMediaPool()
            timeline = project.GetCurrentTimeline()
            if not timeline:
                logger.warning("No current timeline, attempting to find an alternative")
                if project.GetTimelineCount() > 0:
                    logger.debug(f"Project has {project.GetTimelineCount()} timelines, using first")
                    timeline = project.GetTimelineByIndex(1)
                    project.SetCurrentTimeline(timeline)
                    logger.debug(f"Set timeline: {timeline.GetName()}")
                if not timeline:
                    logger.error("No timeline found")
                    self.show_error("No timeline found")
                    return
            else:
                logger.debug(f"Current timeline found: {timeline.GetName()}")
            
            # Check for template
            if not self.template_text:
                logger.error("No Text+ template selected")
                self.show_error("No Text+ template selected")
                return
            
            # Get frame rate
            logger.debug("Opening edit page and getting frame rate")
            resolve.OpenPage("edit")
            frame_rate = int(timeline.GetSetting("timelineFrameRate"))
            if frame_rate == 29:
                frame_rate = 30
            logger.info(f"Using frame rate: {frame_rate}")
            
            # Convert delay to frames
            delay_frames = int(subtitle_delay_seconds * frame_rate)
            logger.info(f"Subtitle delay: {subtitle_delay_seconds}s = {delay_frames} frames")
            
            logger.info(f"Using processed SRT file: {self.processed_srt_path}")
            
            # Parse SRT file
            subs = []
            try:
                logger.debug("Opening SRT file for parsing")
                with open(self.processed_srt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                logger.debug(f"Read {len(lines)} lines from SRT file")
                
                i = 0
                while i < len(lines):
                    try:
                        # Skip empty lines
                        while i < len(lines) and not lines[i].strip():
                            i += 1
                        
                        if i >= len(lines):
                            break
                            
                        # Find index line (a number)
                        if not lines[i].strip().isdigit():
                            logger.debug(f"Line {i+1} is not a subtitle index: {lines[i].strip()}")
                            i += 1
                            continue
                        
                        subtitle_index = lines[i].strip()
                        logger.debug(f"Found subtitle index: {subtitle_index}")
                        
                        # Get timestamp line
                        i += 1
                        if i >= len(lines):
                            logger.warning(f"Unexpected end of file after subtitle index {subtitle_index}")
                            break
                            
                        timestamp_line = lines[i].strip()
                        if " --> " not in timestamp_line:
                            logger.warning(f"Invalid timestamp format for subtitle {subtitle_index}: {timestamp_line}")
                            i += 1
                            continue
                        
                        logger.debug(f"Timestamp line: {timestamp_line}")
                        start_time, end_time = timestamp_line.split(" --> ")
                        
                        # Get text (could be multiple lines)
                        i += 1
                        text_lines = []
                        while i < len(lines) and lines[i].strip():
                            text_lines.append(lines[i].strip())
                            i += 1
                        
                        text = " ".join(text_lines)
                        logger.debug(f"Subtitle text: {text}")
                        
                        # Convert to frames
                        start_frames = self.time_to_frames(start_time, frame_rate)
                        end_frames = self.time_to_frames(end_time, frame_rate)
                        
                        # Apply delay to both start and end times
                        start_frames_delayed = start_frames + delay_frames
                        end_frames_delayed = end_frames + delay_frames
                        
                        # Calculate position and duration - use exact frame positions from SRT with delay
                        timeline_pos = timeline.GetStartFrame() + start_frames_delayed
                        duration = end_frames_delayed - start_frames_delayed
                        
                        # Ensure minimum duration
                        if duration < 1:
                            logger.warning(f"Duration too short for subtitle {subtitle_index}, setting to 1 frame")
                            duration = 1
                            
                        logger.debug(f"Subtitle {subtitle_index}: Original start={start_frames}, Delayed start={start_frames_delayed}, Position={timeline_pos}, Duration={duration}, Text={text}")
                        subs.append((timeline_pos, duration, text))
                    except Exception as e:
                        logger.error(f"Error processing subtitle at line {i}: {str(e)}", exc_info=True)
                        i += 1
                        
                if not subs:
                    logger.error("No valid subtitles found in SRT file")
                    self.show_error("No valid subtitles found in SRT file")
                    return
                
                logger.info(f"Successfully parsed {len(subs)} subtitles with {subtitle_delay_seconds}s delay")
                    
            except Exception as e:
                logger.error(f"Error parsing SRT file: {str(e)}", exc_info=True)
                self.show_error(f"Error parsing SRT file: {str(e)}")
                return
            
            # Assign subtitles to tracks
            logger.debug("Assigning subtitles to tracks")
            track_assignments, assigned_tracks = self.assign_tracks_to_subtitles(subs)
            logger.info(f"Assigned subtitles to {len(set(assigned_tracks))} tracks")
            
            # Prepare timeline tracks
            logger.debug("Preparing timeline tracks")
            current_track_count = timeline.GetTrackCount("video")
            logger.debug(f"Current video track count: {current_track_count}")
            
            timeline.AddTrack("video")
            logger.debug("Added one video track")
            
            new_base_track = current_track_count + 1    
            logger.info(f"New base track index: {new_base_track}")
            self.new_base_track = new_base_track  # Store for reference in other methods
            
            # Adjust track assignments to actual timeline tracks
            for i in range(len(assigned_tracks)):
                assigned_tracks[i] = new_base_track + assigned_tracks[i] - 1
            
            max_assigned_track = max(assigned_tracks)
            logger.debug(f"Maximum assigned track: {max_assigned_track}")
            
            # Add more tracks if needed
            tracks_to_add = max_assigned_track - timeline.GetTrackCount("video")
            if tracks_to_add > 0:
                logger.debug(f"Adding {tracks_to_add} more tracks")
                for _ in range(tracks_to_add):
                    timeline.AddTrack("video")
            
            # Add subtitle clips to timeline
            subtitle_clips = []
            for i, (timeline_pos, duration, text) in enumerate(subs):
                track_num = assigned_tracks[i]
                
                newClip = {
                    "mediaPoolItem": self.template_text,
                    "startFrame": 0,
                    "endFrame": duration,
                    "trackIndex": track_num,
                    "recordFrame": timeline_pos
                }
                subtitle_clips.append(newClip)
                logger.debug(f"Created clip definition {i+1}/{len(subs)}: " +
                        f"Track={track_num}, Start={timeline_pos}, Duration={duration}")
            
            logger.info(f"Attempting to add {len(subtitle_clips)} clips to timeline")
            success = media_pool.AppendToTimeline(subtitle_clips)
            
            if not success:
                logger.error("Failed to add clips to timeline")
                self.show_error("Failed to add clips to timeline")
                return
            
            logger.info("Successfully added clips to timeline")

            # Organize subtitles by track
            subs_by_track = {}
            for idx, (timeline_pos, duration, text) in enumerate(subs):
                track_num = assigned_tracks[idx]
                if track_num not in subs_by_track:
                    subs_by_track[track_num] = []
                subs_by_track[track_num].append(text)

            # Initialize tracking for text similarity and size variation
            prev_texts_by_track = {}
            current_pattern_index_by_track = {}
            size_pattern = [1.4, 0.7]  # Alternating size multipliers (40% bigger, 30% smaller)
            
            logger.info("Updating subtitle text...")
            for track_num, texts in sorted(subs_by_track.items()):
                try:
                    logger.debug(f"Processing track {track_num} with {len(texts)} subtitles")
                    
                    # Initialize for this track if not already done
                    if track_num not in prev_texts_by_track:
                        prev_texts_by_track[track_num] = []
                    if track_num not in current_pattern_index_by_track:
                        current_pattern_index_by_track[track_num] = 0
                        
                    sub_list = timeline.GetItemListInTrack('video', track_num)
                    if not sub_list:
                        logger.warning(f"No items found in track {track_num}")
                        continue
                        
                    logger.debug(f"Found {len(sub_list)} items in track {track_num}")
                    sub_list.sort(key=lambda clip: clip.GetStart())
                    
                    for i, clip in enumerate(sub_list):
                        if i < len(texts):
                            logger.debug(f"Processing clip {i+1}/{len(sub_list)} in track {track_num}")
                            clip.SetClipColor('Orange')
                            text = texts[i]
                            
                            # Check for similarity with previous subtitles
                            is_similar = False
                            for j, prev_text in enumerate(prev_texts_by_track[track_num][-3:]):  # Check last 3 subtitles
                                similarity = self.calculate_text_similarity(text, prev_text)
                                logger.debug(f"Similarity with previous subtitle -{j}: {similarity}")
                                if similarity >= 0.7:  # 70% similarity threshold
                                    is_similar = True
                                    logger.debug(f"Subtitle is similar to a previous one")
                                    break
                            
                            # Add this text to previous texts for future comparisons
                            prev_texts_by_track[track_num].append(text)
                            
                            # Format text
                            max_length = 18
                            max_size = 0.12
                            words = text.split()
                            current_line = ""
                            lines_formatted = []
                            
                            for word in words:
                                if len(current_line) + len(word) + 1 <= max_length:
                                    current_line += word + " "
                                else:
                                    lines_formatted.append(current_line.strip())
                                    current_line = word + " "
                            if current_line:
                                lines_formatted.append(current_line.strip())
                            
                            logger.debug(f"Formatted text into {len(lines_formatted)} lines")
                            
                            # Calculate text size
                            char_count = len(text.replace(" ", ""))
                            starting_size = 0.08
                            size_increase = max(0, 6 - char_count) * 0.1
                            new_size = min(starting_size + size_increase, max_size)
                            logger.debug(f"Base size calculation: chars={char_count}, starting={starting_size}, " +
                                    f"increase={size_increase}, new_size={new_size}")
                            
                            # Apply size variation for similar subtitles
                            if is_similar:
                                current_idx = current_pattern_index_by_track[track_num]
                                size_multiplier = size_pattern[current_idx]
                                new_size = new_size * size_multiplier
                                logger.debug(f"Applying size multiplier for similar subtitle: {size_multiplier}")
                                # Update pattern index for next similar subtitle
                                current_pattern_index_by_track[track_num] = (current_idx + 1) % len(size_pattern)
                            else:
                                # Reset pattern when similarity chain breaks
                                current_pattern_index_by_track[track_num] = 0
                            
                            # Ensure size stays within reasonable bounds, but allow for more variation
                            new_size = max(min(new_size, 0.18), 0.05)
                            logger.debug(f"Final size after adjustments: {new_size}")
                            
                            # Update text in Fusion composition
                            logger.debug("Updating Fusion composition")
                            comp = clip.GetFusionCompByIndex(1)
                            if comp is not None:
                                logger.debug("Got Fusion composition")
                                tools = comp.GetToolList()
                                logger.debug(f"Composition has {len(tools)} tools")
                                
                                for tool_id, tool in tools.items():
                                    tool_name = tool.GetAttrs()['TOOLS_Name']
                                    logger.debug(f"Checking tool: {tool_name}")
                                    
                                    if tool_name == 'Template':
                                        logger.debug(f"Found Template tool (ID: {tool_id})")
                                        comp.SetActiveTool(tool)
                                        
                                        # Instead of using odd/even track numbers, use positions relative to the base track
                                        # This ensures we only have two positions regardless of how many tracks we start with
                                        is_upper_position = (track_num - new_base_track) % 2 != 0
                                        
                                        if is_upper_position:
                                            # Upper position - add newlines to move it up
                                            logger.debug("Using upper position (adding newlines)")
                                            tool.SetInput('StyledText', "\n\n" + text)
                                        else:
                                            # Lower position - no newlines needed
                                            logger.debug("Using lower position (no newlines)")
                                            tool.SetInput('StyledText', text)
                                            
                                        tool.SetInput('Size', new_size)
                                        logger.debug(f"Text and size set successfully")
                            else:
                                logger.warning(f"No Fusion composition found for clip {i}")
                            
                            clip.SetClipColor('Teal')
                            logger.debug(f"Clip {i+1} updated successfully")
                except Exception as e:
                    logger.error(f"Error updating subtitles in track {track_num}: {str(e)}", exc_info=True)
            
            # Save project and show success message
            logger.info("Saving project...")
            project_manager.SaveProject()
            
            # Fixed elapsed time calculation
            try:
                process_end_time = datetime.datetime.now()
                elapsed_time = (process_end_time - process_start_time).total_seconds()
                logger.info(f"Process completed in {elapsed_time:.2f} seconds")
            except Exception as e:
                logger.warning(f"Could not calculate elapsed time: {str(e)}", exc_info=True)
            
            delay_text = f" (with {subtitle_delay_seconds}s delay)" if subtitle_delay_seconds > 0 else ""
            self.status_label.config(text=f"Subtitles added on tracks {new_base_track}-{max_assigned_track}{delay_text}!")
            messagebox.showinfo("Success", f"Added {len(subtitle_clips)} subtitle clips to timeline{delay_text}!")
            
        except Exception as e:
            logger.error(f"Error in add_subtitles: {str(e)}", exc_info=True)
            self.show_error(f"Error adding subtitles: {str(e)}")

    def remove_selected_files(self):
        """Remove selected files from the list"""
        logger.debug("Removing selected files")
        selected_items = self.files_tree.selection()
        
        if not selected_items:
            logger.debug("No files selected for removal")
            return
        
        # Get indices to remove (in reverse order to avoid index shifting)
        indices_to_remove = []
        for item in selected_items:
            index = self.files_tree.index(item)
            indices_to_remove.append(index)
        
        # Remove in reverse order
        for index in sorted(indices_to_remove, reverse=True):
            if 0 <= index < len(self.file_template_pairs):
                removed_file = self.file_template_pairs[index][0]
                logger.debug(f"Removing file: {os.path.basename(removed_file)}")
                del self.file_template_pairs[index]
        
        self.update_files_display()
        self.update_ui_state()

    def select_multiple_srts(self):
        """Open file dialog to select multiple SRT files and assign default templates"""
        logger.debug("Opening file dialog to select multiple SRT files")
        file_paths = filedialog.askopenfilenames(
            title="Select SRT Files",
            filetypes=[("SRT Files", "*.srt"), ("All Files", "*.*")]
        )
        
        if file_paths:
            logger.info(f"Selected {len(file_paths)} SRT files")
            
            # Get default template (first one available)
            default_template = self.mediaPoolItemsList[0] if self.mediaPoolItemsList else None
            
            # Add new files with default template
            for file_path in file_paths:
                logger.debug(f"Adding file: {file_path}")
                # Check if file is already added
                existing_files = [pair[0] for pair in self.file_template_pairs]
                if file_path not in existing_files:
                    self.file_template_pairs.append((file_path, default_template))
                    logger.debug(f"  - Added with default template")
                else:
                    logger.debug(f"  - File already exists, skipping")
            
            self.update_files_display()
            self.update_ui_state()
        else:
            logger.debug("No SRT files selected")

    def recursive_search(self, folder):
        """Recursively search for Text+ templates in media pool folders"""
        folder_name = folder.GetName()
        logger.debug(f"Searching folder: {folder_name}")
        try:
            items = folder.GetClipList()
            logger.debug(f"Found {len(items)} items in folder {folder_name}")
            for item in items:
                try:
                    item_props = item.GetClipProperty()
                    item_type = item_props["Type"]
                    item_name = item_props["Clip Name"]
                    
                    logger.debug(f"Checking item: {item_name}, Type: {item_type}")
                    if item_type == "Fusion Title":
                        logger.info(f"Found Fusion Title: {item_name}")
                        self.mediaPoolItemsList.append(item)
                except Exception as e:
                    logger.error(f"Error processing media pool item: {str(e)}")
            
            # Search subfolders
            subfolders = folder.GetSubFolderList()
            logger.debug(f"Found {len(subfolders)} subfolders in {folder_name}")
            for subfolder in subfolders:
                self.recursive_search(subfolder)
        except Exception as e:
            logger.error(f"Error searching media pool folder {folder_name}: {str(e)}")
    
    def time_to_frames(self, time_str, frame_rate):
        """Convert SRT timestamp to frames - preserves exact timing from SRT"""
        logger.debug(f"Converting time {time_str} to frames at frame rate {frame_rate}")
        hours, minutes, seconds_milliseconds = time_str.split(':')
        seconds, milliseconds = seconds_milliseconds.split(',')
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
        frame_count = int(total_seconds * frame_rate)
        logger.debug(f"Time {time_str} = {total_seconds} seconds = {frame_count} frames")
        # Return exact frame count without adding timeline start offset
        return frame_count

    def clean_srt_file(self, input_srt_path, output_srt_path):
        """Cleans an SRT file by ensuring proper structure"""
        logger.info(f"Cleaning SRT file from {input_srt_path} to {output_srt_path}")
        
        logger.debug("Reading input file")
        with open(input_srt_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        logger.debug(f"Read {len(lines)} lines from file")
        
        cleaned_subs = []
        current_sub = []
        
        for i, line in enumerate(lines):
            if line.strip():
                current_sub.append(line)
                logger.debug(f"Line {i+1} added to current subtitle: {line.strip()}")
            else:
                if len(current_sub) > 2:
                    logger.debug(f"Adding completed subtitle with {len(current_sub)} lines")
                    cleaned_subs.append(current_sub)
                else:
                    logger.debug(f"Skipping incomplete subtitle with only {len(current_sub)} lines")
                current_sub = []
        
        if len(current_sub) > 2:
            logger.debug(f"Adding final subtitle with {len(current_sub)} lines")
            cleaned_subs.append(current_sub)
        
        logger.info(f"Cleaned {len(cleaned_subs)} subtitles")
        logger.debug(f"Writing to {output_srt_path}")
        
        with open(output_srt_path, 'w', encoding='utf-8') as file:
            for i, sub in enumerate(cleaned_subs, start=1):
                sub[0] = f"{i}\n"
                file.writelines(sub)
                file.write('\n')
        
        logger.debug(f"Successfully wrote cleaned SRT file with {len(cleaned_subs)} subtitles")

    def replace_first_commas_with_periods(self, input_file, output_file):
        """Replace every FIRST comma with a period in subtitles that have multiple commas."""
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Replacing first commas with periods from {input_file} to {output_file}")
        
        try:
            # Read and parse SRT file
            subtitles = self._parse_srt_file(input_file)
            logger.debug(f"Loaded {len(subtitles)} subtitles")
            
            # Process each subtitle
            for i, subtitle in enumerate(subtitles):
                content = " ".join(subtitle["content"])
                logger.debug(f"Processing subtitle {i+1}/{len(subtitles)}: {content}")
                
                # Count commas in the text
                comma_count = content.count(',')
                
                if comma_count >= 2:  # Only process if there are at least 2 commas
                    logger.debug(f"Found {comma_count} commas, processing FIRST comma replacement")
                    
                    # Find the position of the FIRST comma
                    first_comma_pos = content.find(',')
                    
                    if first_comma_pos != -1:
                        logger.debug(f"Replacing FIRST comma at position {first_comma_pos} with period")
                        
                        # Replace only the first comma with a period
                        modified_content = content[:first_comma_pos] + '.' + content[first_comma_pos+1:]
                        
                        logger.debug(f"BEFORE: '{content}'")
                        logger.debug(f"AFTER:  '{modified_content}'")
                        
                        # Update subtitle content
                        subtitle["content"] = [modified_content]
                    else:
                        logger.debug("No comma found despite comma count > 0")
                else:
                    logger.debug(f"Only {comma_count} comma(s) found, no replacement needed")
            
            # Write the modified SRT file
            self._write_srt_file(subtitles, output_file)
            logger.info(f"Successfully processed {len(subtitles)} subtitles, replacing first commas with periods")
            return True
            
        except Exception as e:
            logger.error(f"Error replacing first commas with periods: {str(e)}")
            logger.exception(e)
            return False

    def show_error(self, message):
        """Show error message"""
        logger.error(f"ERROR: {message}")
        self.status_label.config(text=f"Error: {message}")
        messagebox.showerror("Error", message)

    def add_commas_before_interjections(self, input_file, output_file):
        """Add commas before interjections in an SRT file - only for mid-sentence interjections."""    
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Adding commas before interjections from {input_file} to {output_file}")
        
        # Define general interjection words (that can appear anywhere in a sentence)
        # These are words that should have a comma BEFORE them when they appear mid-sentence
        general_interjections = [
            # Coordinating conjunctions that need commas when mid-sentence
            'but', 'yet',
            
            # Subordinating conjunctions that often need commas
            'although', 'though', 'even though', 'whereas', 'while', 'unless', 
            'because', 'since', 'as', 'when', 'until', 'before', 'after', 'if',
            
            # Conjunctive adverbs that need commas
            'however', 'nevertheless', 'meanwhile', 'otherwise', 'instead', 'rather',
            'anyway', 'besides', 'moreover', 'furthermore', 'therefore', 'thus',
            'consequently', 'hence', 'accordingly', 'then',
            
            # Transitional phrases that need commas
            'unfortunately', 'fortunately', 'surprisingly', 'subsequently',
            'previously', 'eventually', 'ultimately', 'specifically', 'generally',
            'simultaneously', 'conversely', 'alternatively', 'similarly', 'likewise',
            'in addition', 'in fact', 'in other words', 'in conclusion', 'as a result',
            'in contrast', 'on the other hand', 'for example', 'for instance',
            'that is', 'namely', 'indeed', 'certainly', 'obviously', 'clearly',
            'of course', 'in particular', 'finally', 'lastly',
            
            # Intensifiers that sometimes need commas (use sparingly)
            'actually', 'basically', 'essentially', 'frankly', 'honestly', 'truly',
            'literally', 'seriously', 'absolutely', 'totally', 'completely',
            'utterly', 'precisely', 'really',
            
            # Discourse markers that need commas
            'like', 'you know', 'I mean', 'sort of', 'kind of', 'I guess',
            "let's see", 'how should I put it', 'let me think', "what's the word",
            'if you will', 'in a manner of speaking', 'so to speak', 'as it were'
        ]
        logger.debug(f"Using {len(general_interjections)} general interjection words/phrases")
        logger.debug(f"Full interjection list: {general_interjections}")
        
        # Escape special characters for regex and join with pipe (|)
        general_pattern_str = '|'.join([re.escape(word) for word in general_interjections])
        
        # Create regex pattern for general interjections in mid-sentence
        # This matches when there's no punctuation before the interjection
        pattern = rf'(?<![,.;:!?])\s+\b({general_pattern_str})\b(\s+|$)'
        
        logger.debug(f"FULL REGEX PATTERN: {pattern}")
        
        try:
            logger.debug(f"Opening SRT file: {input_file}")
            
            # Read the SRT file
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse SRT content manually
            subtitles = []
            current_subtitle = {"index": None, "timestamps": None, "content": []}
            
            for line in lines:
                line = line.strip()
                
                if not line:  # Empty line indicates end of a subtitle
                    if current_subtitle["index"] is not None:
                        subtitles.append(current_subtitle)
                        current_subtitle = {"index": None, "timestamps": None, "content": []}
                    continue
                
                if current_subtitle["index"] is None:
                    # This is the subtitle index
                    current_subtitle["index"] = line
                elif current_subtitle["timestamps"] is None:
                    # This is the timestamp line
                    current_subtitle["timestamps"] = line
                else:
                    # This is content
                    current_subtitle["content"].append(line)
            
            # Don't forget the last subtitle if file doesn't end with an empty line
            if current_subtitle["index"] is not None:
                subtitles.append(current_subtitle)
            
            logger.debug(f"Loaded {len(subtitles)} subtitles")
            
            # Process each subtitle
            for i, subtitle in enumerate(subtitles):
                # Join content lines into a single string
                content = " ".join(subtitle["content"])
                
                logger.debug(f"==========================================")
                logger.debug(f"Processing subtitle {i+1}/{len(subtitles)}: {content}")
                logger.debug(f"BEFORE PROCESSING: '{content}'")
                
                # Create protection patterns for each interjection
                # This avoids adding commas when there's already punctuation
                for word in general_interjections:
                    protection_pattern = rf'([,.;:!?])\s+({re.escape(word)})\b'
                    logger.debug(f"Protection pattern for '{word}': {protection_pattern}")
                
                text = content
                
                logger.debug(f"===== Comma Insertion Debug =====")
                logger.debug(f"Original Text after protection: '{text}'")
                words = text.split()
                logger.debug(f"Words in subtitle: {words}")
                
                # Log interjection detection for each word
                for j, word in enumerate(words):
                    cleaned_word = re.sub(r'[^\w]', '', word.lower())
                    is_interjection = cleaned_word in [re.sub(r'[^\w]', '', w.lower()) for w in general_interjections]
                    logger.debug(f"Word {j+1}: '{word}' (cleaned: '{cleaned_word}') - Is interjection? {is_interjection}")
                
                # Find all matches for comma insertion
                matches = list(re.finditer(pattern, text))
                
                logger.debug(f"Found {len(matches)} regex matches for comma insertion")
                for j, match in enumerate(matches):
                    context_start = max(0, match.start() - 5)
                    context_end = min(len(text), match.end() + 15)
                    context = text[context_start:match.start()] + '|' + text[match.start():match.end()] + '|' + text[match.end():context_end]
                    context = context.replace('\n', ' ')
                    
                    logger.debug(f"Match {j+1}: '{match.group()}' at positions ({match.start()}, {match.end()})")
                    logger.debug(f"  Context: '...{context}...'")
                    logger.debug(f"  Captured groups: {match.groups()}")
                
                # Apply comma insertion
                modified_text = re.sub(pattern, r', \1\2', text)
                
                logger.debug(f"Modified Text: '{modified_text}'")
                logger.debug(f"Regex Pattern: {pattern}")
                
                # Check if there were changes
                if modified_text != text:
                    logger.debug(f"CHANGES DETECTED - Added commas before interjections:")
                    logger.debug(f"  BEFORE: '{text}'")
                    logger.debug(f"  AFTER:  '{modified_text}'")
                    
                    # Log word-by-word differences
                    orig_words = text.split()
                    new_words = modified_text.split()
                    for j in range(min(len(orig_words), len(new_words))):
                        if orig_words[j] != new_words[j]:
                            logger.debug(f"  Difference at word {j+1}:")
                            logger.debug(f"    Original word: '{orig_words[j]}'")
                            logger.debug(f"    Modified word: '{new_words[j]}'")
                
                # Update subtitle content
                subtitle["content"] = [modified_text]
                
                logger.debug(f"Final subtitle text: '{modified_text}'")
                logger.debug(f"==========================================")
            
            # Write the modified SRT file
            logger.debug(f"Saving processed subtitles to {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, subtitle in enumerate(subtitles):
                    # Write index
                    f.write(f"{subtitle['index']}\n")
                    
                    # Write timestamps
                    f.write(f"{subtitle['timestamps']}\n")
                    
                    # Write content
                    for line in subtitle["content"]:
                        f.write(f"{line}\n")
                    
                    # Add blank line between subtitles (except after the last one)
                    if i < len(subtitles) - 1:
                        f.write("\n")
            
            logger.info(f"Successfully processed {len(subtitles)} subtitles, adding commas before interjections")
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding commas before interjections: {str(e)}")
            logger.exception(e)
            return False

    def add_commas_after_starting_interjections(self, srt_file, output_file):
        """Add commas after interjections that appear at the beginning of a sentence."""
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Adding commas after starting interjections from {srt_file} to {output_file}")
        
        # Complete list of starting interjections
        starting_interjections = [
            # True interjections and exclamations
            'well', 'oh', 'ah', 'yeah', 'agh', 'ugh', 'um', 'uh', 'er', 'hmm',
            'ahem', 'wow', 'ouch', 'oops', 'yikes', 'gosh', 'geez', 'dang', 'darn',
            'hey', 'hi', 'hello', 'whoa', 'phew', 'huh', 'yay', 'boo', 'aw', 'eek',
            'psst', 'shh', 'yep', 'nope', 'mhm', 'eh', 'ooh', 'aah', 'oof',
            'damn', 'jeez',
            
            # Discourse markers that start sentences
            'actually', 'basically', 'essentially', 'frankly', 'honestly', 'truly',
            'look', 'listen', 'see', 'okay', 'alright', 'right',
            
            # Sequence markers
            'first', 'second', 'third', 'next', 'then', 'finally', 'lastly',
            'initially', 'subsequently', 'ultimately',
            
            # Conjunctive adverbs at sentence start
            'however', 'nevertheless', 'meanwhile', 'otherwise', 'instead', 'rather',
            'anyway', 'besides', 'moreover', 'furthermore', 'therefore', 'thus',
            'consequently', 'hence', 'accordingly',
            
            # Transitional phrases at sentence start
            'unfortunately', 'fortunately', 'surprisingly', 'previously', 'eventually',
            'specifically', 'generally', 'simultaneously', 'conversely', 'alternatively',
            'similarly', 'likewise', 'in addition', 'in fact', 'in other words',
            'in conclusion', 'as a result', 'in contrast', 'on the other hand',
            'for example', 'for instance', 'that is', 'namely', 'indeed', 'certainly',
            'obviously', 'clearly', 'of course', 'in particular', 'after that',
            'before that',
            
            # Time and condition markers
            'when', 'while', 'since', 'until', 'after', 'before', 'unless', 'if',
            
            # Qualifying phrases
            'you know', 'I mean', 'sort of', 'kind of', 'I guess', "let's see",
            'how should I put it', 'let me think', "what's the word", 'if you will',
            'in a manner of speaking', 'so to speak', 'as it were',
            
            # Emphatic words at sentence start
            'literally', 'seriously', 'absolutely', 'totally', 'completely',
            'utterly', 'precisely', 'really'
        ]
        
        logger.debug(f"Using {len(starting_interjections)} starting interjection words/phrases")
        
        try:
            # Parse SRT file
            subtitles = self._parse_srt_file(srt_file)
            logger.debug(f"Loaded {len(subtitles)} subtitles")
            
            # Create TWO regex patterns for comprehensive coverage
            
            # Pattern 1: For interjections followed by whitespace (original pattern)
            pattern1 = r'(^|[.!?]\s+)\b('
            pattern1 += '|'.join(map(re.escape, starting_interjections))
            pattern1 += r')\b\s+(?!,)'  # Negative lookahead to avoid adding comma if already exists
            
            # Pattern 2: NEW - For interjections followed by punctuation then whitespace (like "oh. Jesus")
            pattern2 = r'(^|[.!?]\s+)\b('
            pattern2 += '|'.join(map(re.escape, starting_interjections))
            pattern2 += r')\b([.!?])\s+'  # Capture the punctuation after the interjection
            
            logger.debug(f"Using regex pattern 1 for starting interjections: {pattern1}")
            logger.debug(f"Using regex pattern 2 for punctuated interjections: {pattern2}")
            
            # Process each subtitle
            for sub_index, subtitle in enumerate(subtitles):
                content = " ".join(subtitle["content"])
                original_content = content
                
                logger.debug(f"Processing subtitle {sub_index+1}/{len(subtitles)}: {content}")
                
                # First, protect interjections that already have commas
                for interjection in starting_interjections:
                    protection_pattern = r'(^|[.!?]\s+)(' + re.escape(interjection) + r'),\s+'
                    content = re.sub(protection_pattern, r'\1\2<<PROTECTED>>,', content, flags=re.IGNORECASE)
                
                # Apply Pattern 1: Standard interjections followed by whitespace
                content = re.sub(pattern1, r'\1\2, ', content, flags=re.IGNORECASE)
                logger.debug(f"After pattern 1: '{content}'")
                
                # Apply Pattern 2: Interjections followed by punctuation then whitespace
                content = re.sub(pattern2, r'\1\2,\3 ', content, flags=re.IGNORECASE)
                logger.debug(f"After pattern 2: '{content}'")
                
                # Clean up awkward punctuation combinations like ",." -> ","
                content = re.sub(r',([.!?])\s+', r', ', content)
                logger.debug(f"After punctuation cleanup: '{content}'")
                
                # Restore protected instances
                content = content.replace('<<PROTECTED>>,', ',')
                
                if content != original_content:
                    logger.debug(f"Enhanced interjection processing:")
                    logger.debug(f"  BEFORE: '{original_content}'")
                    logger.debug(f"  AFTER:  '{content}'")
                    subtitle["content"] = [content]
            
            # Write the modified SRT file
            self._write_srt_file(subtitles, output_file)
            logger.info(f"Successfully processed {len(subtitles)} subtitles with enhanced interjection handling")
            return True
            
        except Exception as e:
            logger.error(f"Error adding commas after starting interjections: {str(e)}")
            logger.exception(e)
            return False

    def split_subtitles_at_commas(self, srt_file, output_file):
        """Split subtitles at commas, with controlled overlaps between pairs only"""
        logger.info(f"Splitting subtitles at commas from {srt_file} to {output_file}")
        
        subs = pysrt.open(srt_file, encoding='utf-8')
        logger.debug(f"Loaded {len(subs)} subtitles")
        
        new_subs = pysrt.SubRipFile()
        counter = 1
        
        # Track original subtitle boundaries to prevent cross-boundary overlaps
        original_subtitle_groups = []  # List of lists - each inner list contains subtitle indices from same original
        
        for sub_index, sub in enumerate(subs):
            # Skip empty subtitles
            if not sub.text.strip():
                logger.debug(f"Skipping empty subtitle {sub_index+1}")
                continue
            
            logger.debug(f"Processing subtitle {sub_index+1}: {sub.text}")
            
            # Check if there are any commas in the text
            if ',' not in sub.text:
                logger.debug(f"No commas in subtitle {sub_index+1}, keeping as is")
                new_subs.append(pysrt.SubRipItem(
                    index=counter,
                    start=sub.start,
                    end=sub.end,
                    text=sub.text.strip()
                ))
                # Single subtitle forms its own group
                original_subtitle_groups.append([counter - 1])  # 0-based index
                counter += 1
                continue
            
            # Split the subtitle text at commas and filter out empty segments
            segments = [seg.strip() for seg in sub.text.split(',') if seg.strip()]
            logger.debug(f"Split into {len(segments)} segments at commas: {segments}")
            
            # If there's only one meaningful segment after splitting, keep as is
            if len(segments) <= 1:
                logger.debug(f"Only one meaningful segment after comma split, keeping as is")
                new_subs.append(pysrt.SubRipItem(
                    index=counter,
                    start=sub.start,
                    end=sub.end,
                    text=sub.text.strip()
                ))
                original_subtitle_groups.append([counter - 1])
                counter += 1
                continue
            
            # Calculate total duration and start/end times
            start_ms = sub.start.ordinal
            end_ms = sub.end.ordinal
            total_duration = end_ms - start_ms
            logger.debug(f"Total duration: {total_duration} ms, Start: {start_ms}, End: {end_ms}")
            
            # Track which new subtitles belong to this original subtitle
            current_group = []
            
            # Calculate segment durations based on character length
            segment_durations = []
            total_chars = sum(len(seg) for seg in segments)
            logger.debug(f"Total characters: {total_chars}")
            
            for seg in segments:
                duration = max(300, int((len(seg) / total_chars) * total_duration))
                segment_durations.append(duration)
                logger.debug(f"Segment '{seg}' has {len(seg)} chars, calculated duration: {duration} ms")
            
            # Adjust durations to ensure they fit within the total subtitle time
            total_allocated = sum(segment_durations)
            if total_allocated > total_duration:
                scale_factor = total_duration / total_allocated
                original_durations = segment_durations.copy()
                segment_durations = [int(duration * scale_factor) for duration in segment_durations]
                logger.debug(f"Total allocated time ({total_allocated} ms) exceeds available time ({total_duration} ms)")
                logger.debug(f"Adjusted durations with scale factor {scale_factor}")
                for i, (orig, adjusted) in enumerate(zip(original_durations, segment_durations)):
                    logger.debug(f"  Segment {i+1}: {orig} ms -> {adjusted} ms")
            
            # Process segments with controlled overlaps
            segment_start = start_ms
            
            for i, segment in enumerate(segments):
                if i == 0 and len(segments) > 1:
                    segment_text = segment + ','
                    logger.debug(f"Adding comma to first segment: '{segment}' -> '{segment_text}'")
                else:
                    segment_text = segment
                    logger.debug(f"Using clean segment: '{segment_text}'")
                
                # Determine segment duration
                seg_duration = segment_durations[i]
                
                # Calculate end time - base end is always start + duration
                base_end = segment_start + seg_duration
                
                # Only the first segment in a sequence can overlap with the next one
                if i == 0:
                    if len(segments) > 1:
                        next_duration = segment_durations[1]
                        half_next_duration = next_duration // 2
                        extended_end = min(base_end + half_next_duration, end_ms)
                    else:
                        extended_end = base_end
                else:
                    extended_end = base_end
                
                logger.debug(f"Segment {i+1}/{len(segments)}: '{segment_text}'")
                logger.debug(f"  Start: {segment_start} ms, End: {extended_end} ms")
                logger.debug(f"  Duration: {extended_end - segment_start} ms")
                
                # Create the subtitle
                new_subs.append(pysrt.SubRipItem(
                    index=counter,
                    start=pysrt.SubRipTime.from_ordinal(segment_start),
                    end=pysrt.SubRipTime.from_ordinal(extended_end),
                    text=segment_text
                ))
                
                # Add to current group
                current_group.append(counter - 1)  # 0-based index
                counter += 1
                
                if i < len(segments) - 1:
                    segment_start = base_end
            
            # Add this group to the original subtitle groups
            original_subtitle_groups.append(current_group)
        
        # Save the new subtitles file
        logger.info(f"Split subtitles at commas into {len(new_subs)} total subtitles")
        logger.debug(f"Original subtitle groups: {original_subtitle_groups}")
        logger.debug(f"Saving to {output_file}")
        new_subs.save(output_file, encoding='utf-8')
        
        # Store the grouping information for use in overlap fixing
        self._original_subtitle_groups = original_subtitle_groups

    def fix_interjection_overlaps(self, input_file, output_file):
        """
        Fix missing overlaps between short interjections and following content.
        NOW INCLUDES TRACK-BASED LOGIC: Only odd tracks (1,3,5...) can overlap with even tracks (2,4,6...)
        Even tracks should end cleanly without overlapping with the next subtitle.
        """
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Fixing interjection overlaps from {input_file} to {output_file}")
        
        try:
            import pysrt
            
            # Load SRT file
            subs = pysrt.open(input_file, encoding='utf-8')
            logger.debug(f"Loaded {len(subs)} subtitles for overlap fixing")
            
            # Get original subtitle groupings (set during comma splitting)
            original_groups = getattr(self, '_original_subtitle_groups', [])
            logger.debug(f"Original subtitle groups: {original_groups}")
            
            # Create a mapping of subtitle index to group index
            subtitle_to_group = {}
            for group_idx, group in enumerate(original_groups):
                for subtitle_idx in group:
                    subtitle_to_group[subtitle_idx] = group_idx
            
            logger.debug(f"Subtitle to group mapping: {subtitle_to_group}")
            
            # Get track assignments from the track assignment phase
            # We need to determine which track each subtitle was assigned to
            # Based on the assign_tracks_to_subtitles logic: alternating between tracks 1 and 2
            
            # Define interjection patterns that should overlap with following content
            overlap_interjections = [
                'oh', 'ah', 'well', 'yeah', 'yes', 'no', 'but', 'and', 'so', 'now',
                'hey', 'hi', 'wow', 'okay', 'right', 'then', 'plus', 'also',
                'damn', 'shit', 'fuck', 'jesus', 'god', 'christ', 'holy',
                'actually', 'basically', 'honestly', 'look', 'listen', 'see'
            ]
            
            # Content types that benefit from interjection overlaps
            overlap_worthy_content = [
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'that', 'this',
                'a', 'an', 'the',
                'guy', 'man', 'woman', 'person', 'people', 'jesus', 'christ', 'god',
                'there', 'here', 'what', 'where', 'when', 'how', 'why'
            ]
            
            overlap_fixes_made = 0
            
            # Simulate track assignment to determine which track each subtitle is on
            track_assignments = self._simulate_track_assignments(subs)
            logger.debug(f"Simulated track assignments: {track_assignments}")
            
            # Process consecutive subtitle pairs
            for i in range(len(subs) - 1):
                current_sub = subs[i]
                next_sub = subs[i + 1]
                
                logger.debug(f"Analyzing pair {i+1}-{i+2}: '{current_sub.text}' + '{next_sub.text}'")
                
                # NEW: Check track-based overlap rules
                current_track = track_assignments.get(i, 1)
                next_track = track_assignments.get(i + 1, 2)
                
                logger.debug(f"  Track assignment: subtitle {i+1} on track {current_track}, subtitle {i+2} on track {next_track}")
                
                # CRITICAL RULE: Only odd tracks can overlap with even tracks
                # Even tracks should end cleanly without overlapping
                if current_track % 2 == 0:  # Even track (2, 4, 6...)
                    logger.debug(f"  SKIPPING: Current subtitle is on even track {current_track} - no overlap allowed")
                    continue
                
                if next_track % 2 != 0:  # Next track is odd
                    logger.debug(f"  SKIPPING: Next subtitle is on odd track {next_track} - would create odd-to-odd overlap")
                    continue
                
                logger.debug(f"  TRACK RULE PASSED: Odd track {current_track} can overlap with even track {next_track}")
                
                # CHECK: Are these subtitles from the same original subtitle group?
                current_group = subtitle_to_group.get(i)
                next_group = subtitle_to_group.get(i + 1)
                
                if current_group is not None and next_group is not None and current_group != next_group:
                    logger.debug(f"  SKIPPING: Subtitles are from different original groups ({current_group} vs {next_group})")
                    continue
                
                # Check if current subtitle is a short interjection candidate
                if self._is_interjection_candidate(current_sub.text, overlap_interjections):
                    logger.debug(f"  '{current_sub.text}' is an interjection candidate")
                    
                    # Check if next subtitle is overlap-worthy content
                    if self._is_overlap_worthy_content(next_sub.text, overlap_worthy_content):
                        logger.debug(f"  '{next_sub.text}' is overlap-worthy content")
                        
                        # Additional check: Only create overlap within the same original group
                        if current_group == next_group:
                            logger.debug(f"  Both subtitles from same original group ({current_group}) - overlap allowed")
                            
                            # Check timing to see if overlap is missing or insufficient
                            current_end = current_sub.end.ordinal
                            next_start = next_sub.start.ordinal
                            gap_duration = next_start - current_end
                            
                            logger.debug(f"  Timing analysis: gap = {gap_duration}ms")
                            
                            # Create overlap if there's a gap or very small overlap
                            if gap_duration >= -200:  # Gap or overlap less than 200ms
                                overlap_duration = self._calculate_ideal_overlap(
                                    current_sub.text, next_sub.text
                                )
                                
                                if overlap_duration > 0:
                                    # Extend current subtitle to overlap into next
                                    new_current_end = next_start + overlap_duration
                                    
                                    # Don't extend past the next subtitle's end
                                    max_extension = next_sub.end.ordinal - next_start
                                    actual_extension = min(overlap_duration, max_extension - 100)  # Leave 100ms minimum
                                    
                                    if actual_extension > 100:  # Only apply if meaningful overlap
                                        new_current_end = next_start + actual_extension
                                        
                                        logger.info(f"Creating overlap between subtitles {i+1}-{i+2}:")
                                        logger.info(f"  Interjection: '{current_sub.text}' (Track {current_track})")
                                        logger.info(f"  Following: '{next_sub.text}' (Track {next_track})")
                                        logger.info(f"  Original gap: {gap_duration}ms")
                                        logger.info(f"  New overlap: {actual_extension}ms")
                                        
                                        # Update the timing
                                        current_sub.end = pysrt.SubRipTime.from_ordinal(new_current_end)
                                        overlap_fixes_made += 1
                                    else:
                                        logger.debug(f"  Overlap too small ({actual_extension}ms), skipping")
                                else:
                                    logger.debug(f"  No overlap recommended for this pair")
                            else:
                                logger.debug(f"  Adequate overlap already exists ({-gap_duration}ms)")
                        else:
                            logger.debug(f"  Different original groups - no overlap created")
                    else:
                        logger.debug(f"  '{next_sub.text}' is not suitable for overlap")
                else:
                    logger.debug(f"  '{current_sub.text}' is not an interjection candidate")
            
            # Save the modified SRT file
            subs.save(output_file, encoding='utf-8')
            
            logger.info(f"Fixed {overlap_fixes_made} interjection overlaps using track-based rules")
            return True
            
        except Exception as e:
            logger.error(f"Error fixing interjection overlaps: {str(e)}")
            logger.exception(e)
            return False

    def _simulate_track_assignments(self, subs):
        """
        Simulate the track assignment logic to determine which track each subtitle is on.
        This replicates the logic from assign_tracks_to_subtitles.
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        # Convert subtitles to the format expected by assign_tracks_to_subtitles
        subs_data = []
        for sub in subs:
            start = sub.start.ordinal
            duration = sub.end.ordinal - sub.start.ordinal
            text = sub.text
            subs_data.append((start, duration, text))
        
        # Use the same assignment logic as the main function
        track_assignments, assigned_tracks = self.assign_tracks_to_subtitles(subs_data)
        
        # Convert to a simple mapping: subtitle_index -> track_number
        track_mapping = {}
        for i, track in enumerate(assigned_tracks):
            track_mapping[i] = track
        
        logger.debug(f"Track assignment simulation complete: {track_mapping}")
        return track_mapping

    def assign_tracks_to_subtitles(self, subs):
        """
        Assign subtitles to tracks ensuring overlaps only happen in pairs, never allowing
        three subtitles to display simultaneously. Now with special handling for short
        singular words that should go to a 3rd track.
        """
        logger.info(f"Assigning {len(subs)} subtitles to tracks with improved overlap control and 3rd track logic")
        
        # Sort subtitles by start time
        sorted_indices = sorted(range(len(subs)), key=lambda i: subs[i][0])
        track_assignments = {1: [], 2: [], 3: []}  # Add track 3
        assigned_tracks = [None] * len(subs)
        
        # Keep track of the end time of the last subtitle on each track
        last_end_time = {1: 0, 2: 0, 3: 0}
        active_subtitle_on_track = {1: None, 2: None, 3: None}
        
        for idx in sorted_indices:
            start, duration, text = subs[idx]
            end = start + duration
            logger.debug(f"Processing subtitle {idx}: Start={start}, End={end}, Text={text[:20]}...")
            
            # Check if this is a short singular word that should go to track 3
            should_use_track_3 = self._should_use_track_3(subs, idx, sorted_indices)
            
            if should_use_track_3:
                # Special handling for track 3 - extend previous subtitle's duration with limits
                prev_idx = self._get_previous_subtitle_index(idx, sorted_indices)
                if prev_idx is not None:
                    prev_start, prev_duration, prev_text = subs[prev_idx]
                    original_prev_end = prev_start + prev_duration
                    
                    # Calculate maximum allowed extension
                    max_extension = min(
                        1000,  # Maximum 1 second extension
                        prev_duration * 2,  # Don't more than double the original duration
                        end - original_prev_end  # Don't extend beyond track 3 subtitle's end
                    )
                    
                    # Only extend if it's reasonable and beneficial
                    if max_extension > 100 and (end - original_prev_end) <= max_extension:
                        new_prev_duration = end - prev_start
                        subs[prev_idx] = (prev_start, new_prev_duration, prev_text)
                        
                        # Update tracking for the extended subtitle
                        prev_track = assigned_tracks[prev_idx]
                        if prev_track:
                            last_end_time[prev_track] = end
                        
                        logger.info(f"Extended subtitle {prev_idx} ('{prev_text[:20]}...') by {max_extension}ms to accommodate track 3 subtitle {idx} ('{text}')")
                    else:
                        logger.debug(f"Skipping extension for subtitle {prev_idx} - extension would be too large or unreasonable")
                
                # Assign to track 3
                track_assignments[3].append(idx)
                assigned_tracks[idx] = 3
                last_end_time[3] = end
                active_subtitle_on_track[3] = idx
                
                logger.info(f"Assigned short singular subtitle {idx} ('{text}') to track 3")
                continue
            
            # Original track assignment logic for tracks 1 and 2
            # Finding the track with the earliest end time
            min_end_track = 1 if last_end_time[1] <= last_end_time[2] else 2
            
            # Default to track 1 if both tracks are free
            track_to_use = 1 if last_end_time[1] <= start and last_end_time[2] <= start else None
            
            if track_to_use is None:
                # If there's overlap, first check if there's a track that's completely free
                if start >= last_end_time[1]:
                    track_to_use = 1
                    logger.debug(f"  Track 1 is free, using it")
                elif start >= last_end_time[2]:
                    track_to_use = 2
                    logger.debug(f"  Track 2 is free, using it")
                else:
                    # Both tracks have active subtitles, find the one ending soonest
                    track_to_use = min_end_track
                    logger.debug(f"  Both tracks have active subtitles, using track {track_to_use} (ends soonest)")
                    
                    # Critical fix: If we're placing on track with an active subtitle,
                    # extend the virtual end time of the OTHER track to prevent a third subtitle
                    # from appearing before this pair is done
                    other_track = 3 - track_to_use  # If track_to_use is 1, other_track is 2, and vice versa
                    last_end_time[other_track] = max(last_end_time[other_track], end)
                    logger.debug(f"  Extended virtual end time of track {other_track} to {end} to prevent triple overlaps")
            
            # Assign to chosen track
            track_assignments[track_to_use].append(idx)
            assigned_tracks[idx] = track_to_use
            
            # Update the active subtitle and end time for the assigned track
            active_subtitle_on_track[track_to_use] = idx
            last_end_time[track_to_use] = end
            
            logger.debug(f"  Assigned subtitle {idx} to track {track_to_use} (ends at {end})")
        
        # Count usage of each track
        track_1_count = len(track_assignments[1])
        track_2_count = len(track_assignments[2])
        track_3_count = len(track_assignments[3])
        logger.info(f"Subtitles assigned to tracks: Track 1: {track_1_count}, Track 2: {track_2_count}, Track 3: {track_3_count}")
        
        return track_assignments, assigned_tracks

    def _should_use_track_3(self, subs, current_idx, sorted_indices):
        """
        Determine if a subtitle should be placed on track 3.
        Criteria:
        1. Short text (1-2 words only)
        2. Single word that's not an interjection
        3. Very short duration (< 400ms)
        4. Follows closely after previous subtitle (< 200ms gap)
        """
        start, duration, text = subs[current_idx]
        
        # Check if it's a very short text (1-2 words max, be more restrictive)
        words = text.strip().split()
        if len(words) > 2:
            return False
        
        # Must be very short duration
        if duration >= 400:  # Reduced from 500ms
            return False
        
        # Check if it's a single word that's likely not an interjection
        if len(words) == 1:
            word = words[0].lower().strip('.,!?;:')
            
            # Common interjections that should stay in normal flow
            interjections = ['oh', 'ah', 'yeah', 'wow', 'hey', 'hmm', 'uh', 'um', 'well', 'so', 'but', 'and']
            
            if word in interjections:
                return False
            
            # Check if this follows closely after the previous subtitle
            if self._follows_closely(current_idx, sorted_indices, subs):
                logger.debug(f"Subtitle {current_idx} ('{text}') qualifies for track 3: single word '{word}', short duration ({duration}ms), follows closely")
                return True
        
        # Check for very short phrases that are conclusions/endings  
        if len(words) == 2:
            # Only very specific short endings
            short_endings = ['no.', 'yes.', 'ok.', 'fine.', 'done.', 'stop.']
            text_clean = text.lower().strip()
            
            if text_clean in short_endings and self._follows_closely(current_idx, sorted_indices, subs):
                logger.debug(f"Subtitle {current_idx} ('{text}') qualifies for track 3: short ending phrase, follows closely")
                return True
        
        return False

    def _follows_closely(self, current_idx, sorted_indices, subs):
        """
        Check if the current subtitle follows closely after the previous one.
        """
        # Find current position in sorted list
        try:
            current_pos = sorted_indices.index(current_idx)
        except ValueError:
            return False
        
        # Need at least 1 previous subtitle
        if current_pos < 1:
            return False
        
        # Get the previous subtitle
        prev_idx = sorted_indices[current_pos - 1]
        
        # Check timing gap
        prev_start, prev_duration, prev_text = subs[prev_idx]
        current_start, current_duration, current_text = subs[current_idx]
        
        prev_end = prev_start + prev_duration
        gap = current_start - prev_end
        
        # Must follow closely (within 200ms gap) and have overlapping potential
        if gap <= 200 and gap >= -100:  # Small gap or slight overlap
            logger.debug(f"Subtitle {current_idx} follows closely after {prev_idx}: gap = {gap}ms")
            return True
        
        return False

    def _get_previous_subtitle_index(self, current_idx, sorted_indices):
        """
        Get the index of the previous subtitle in the sorted order.
        """
        try:
            current_pos = sorted_indices.index(current_idx)
            if current_pos > 0:
                return sorted_indices[current_pos - 1]
        except (ValueError, IndexError):
            pass
        return None

    def _follows_overlapping_pair(self, current_idx, sorted_indices):
        """
        Check if the current subtitle follows an overlapping pair.
        """
        # Find current position in sorted list
        current_pos = sorted_indices.index(current_idx)
        
        # Need at least 2 previous subtitles to form a pair
        if current_pos < 2:
            return False
        
        # Get the two previous subtitles
        prev_idx = sorted_indices[current_pos - 1]
        prev_prev_idx = sorted_indices[current_pos - 2]
        
        # Check if the previous two subtitles overlap
        # (This is a simplified check - in reality we'd check the actual timing)
        return True  # For now, assume they could overlap if they're consecutive

    def split_long_subtitles(self, input_srt, output_srt, max_length=20):
        """Split long subtitles into smaller segments with improved handling of determiners and short final segments."""
        logger.info(f"Splitting long subtitles from {input_srt} to {output_srt} (max length: {max_length})")
        
        # Words that should not be split after (like abbreviations)
        do_not_split_after = ["Mr.", "Ms.", "Dr.", "Mrs.", "Jr.", "Sr.", "St.", "Co.", "Inc.", "Ltd.", "Gov.", "e.g.", "i.e."]
        
        # Determiners that should always stay with their following noun (never split after these)
        determiners = [
            # Articles
            "a", "an", "the",
            # Demonstrative determiners
            "this", "that", "these", "those",
            # Possessive determiners
            "my", "your", "his", "her", "its", "our", "their", 
            # Quantifiers
            "some", "any", "many", "much", "few", "all", "every", "each", "either", "neither",
            "no", "several", "enough", "various", "certain",
            # Numbers as determiners
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            # Interrogative determiners
            "which", "what", "whose",
            # Other determiners
            "both", "half", "such", "same", "other", "another"
        ]
        
        # Words that can be good split points (like conjunctions)
        conjunctions = ["and", "but", "or", "because", "so", "although", "though"]
        
        logger.debug(f"Words to not split after: {do_not_split_after}")
        logger.debug(f"Determiner words that must stay with their nouns: {determiners}")
        logger.debug(f"Conjunctions that can be split points: {conjunctions}")
        
        logger.debug("Opening SRT file for parsing")
        subs = pysrt.open(input_srt, encoding='utf-8')
        logger.debug(f"Loaded {len(subs)} subtitles from file")
        
        new_subs = pysrt.SubRipFile()
        counter = 1
        
        # Minimum words allowed in a segment to prevent tiny segments
        min_words_per_segment = 2
        logger.debug(f"Minimum words per segment: {min_words_per_segment}")
        
        for sub_index, sub in enumerate(subs):
            logger.debug(f"Processing subtitle {sub_index+1}/{len(subs)}: {sub.text}")
            logger.debug("===== Long Subtitle Splitting Debug =====")
            
            # If subtitle is already short enough, keep it as is
            if len(sub.text) <= max_length:
                logger.debug(f"Subtitle {sub_index+1} is already short enough ({len(sub.text)} chars <= {max_length} chars)")
                new_subs.append(pysrt.SubRipItem(
                    index=counter,
                    start=sub.start,
                    end=sub.end,
                    text=sub.text.strip()
                ))
                counter += 1
                continue
            
            logger.debug(f"Subtitle {sub_index+1} needs splitting ({len(sub.text)} chars > {max_length} chars)")
            words = sub.text.split()
            logger.debug(f"Subtitle contains {len(words)} words: {words}")
            total_duration = sub.end.ordinal - sub.start.ordinal
            logger.debug(f"Total subtitle duration: {total_duration} ms")
            
            # Find all possible good split points
            split_points = []
            current_line = []
            current_length = 0
            
            logger.debug("Analyzing potential split points:")
            for i, word in enumerate(words):
                word_lower = word.lower().strip(',.;:!?')
                
                # Log decision-making for each word
                logger.debug(f"Word {i+1}/{len(words)}: '{word}' (lowercase cleaned: '{word_lower}')")
                
                # Skip checking split points after abbreviations
                if i > 0 and words[i-1] in do_not_split_after:
                    logger.debug(f"  Skipping split check after abbreviation: {words[i-1]}")
                    current_line.append(word)
                    current_length += len(word) + 1  # +1 for the space
                    continue
                
                # NEVER consider adding a split point before a word if the previous word is a determiner
                if i > 0 and words[i-1].lower().strip(',.;:!?') in determiners:
                    logger.debug(f"  NEVER splitting after determiner '{words[i-1]}'")
                    current_line.append(word)
                    current_length += len(word) + 1  # +1 for the space
                    continue
                    
                # Add word to current line
                current_line.append(word)
                current_length += len(word) + 1  # +1 for the space
                logger.debug(f"  Current line length: {current_length} chars, words: {current_line}")
                
                # CRITICAL: Don't add a split point if the next word is a determiner
                # Always keep determiners with the following word
                if i < len(words) - 1 and words[i+1].lower().strip(',.;:!?') in determiners:
                    continue
                
                # Consider this a potential split point if:
                # 1. We're at the preferred length (12-15 chars) AND
                # 2. Either: word ends with punctuation, or it's a conjunction, or we're getting too long (>20)
                is_punctuated = word.endswith((",", ".", ";", ":", "!", "?"))
                is_conjunction = word_lower in conjunctions
                
                if current_length >= 12:  # Good minimum length for a subtitle line
                    logger.debug(f"  Potential split point candidate - Length: {current_length} chars")
                    logger.debug(f"  Word properties - Punctuated: {is_punctuated}, Conjunction: {is_conjunction}")
                    
                    if is_punctuated or is_conjunction or current_length > max_length:
                        logger.debug(f"  Qualifies as split point: punctuated={is_punctuated}, conjunction={is_conjunction}, over max length={current_length > max_length}")
                        
                        # Only add this split point if it wouldn't create a too-short final segment
                        words_remaining = len(words) - (i + 1)
                        logger.debug(f"  Words remaining after this point: {words_remaining}")
                        
                        if words_remaining >= min_words_per_segment or i == len(words) - 1:
                            logger.debug(f"  Adding split point at word {i+1} ('{word}')")
                            split_points.append(i)
                            current_line = []
                            current_length = 0
                        else:
                            logger.debug(f"  Not adding split point - would create a too-short final segment ({words_remaining} < {min_words_per_segment} words)")
                    else:
                        logger.debug(f"  Not adding split point - doesn't qualify (needs punctuation, conjunction, or exceeding max length)")
                else:
                    logger.debug(f"  Not considering split point yet - current line too short ({current_length} < 12 chars)")
            
            logger.debug(f"Initial split points: {split_points}")
            
            # Filter out split points that would separate determiners from their nouns
            filtered_split_points = []
            for i, split_idx in enumerate(split_points):
                # Check if the next word after the split is a determiner
                if split_idx + 1 < len(words) and words[split_idx + 1].lower().strip(',.;:!?') in determiners:
                    logger.debug(f"Removing split point at word {split_idx+1} ('{words[split_idx]}') because next word '{words[split_idx+1]}' is a determiner")
                    continue
                    
                # Check if this would split a determiner from its noun
                if split_idx > 0 and words[split_idx].lower().strip(',.;:!?') in determiners:
                    logger.debug(f"Removing split point at word {split_idx+1} ('{words[split_idx]}') because it's a determiner")
                    continue
                    
                filtered_split_points.append(split_idx)
            
            logger.debug(f"Split points after filtering out determiners: {filtered_split_points}")
            split_points = filtered_split_points
            
            # If no good split points found or very long text, split more aggressively
            # but still respect determiner-noun pairs
            if not split_points or (len(words) > 10 and len(split_points) < len(words) // 5):
                logger.debug("Not enough good split points found, using forced splits")
                # Reset and try again with forced splits but still respect minimum segment size
                split_points = []
                chunk_size = min(4, len(words) // 2)  # Max 4 words per line or half the total
                logger.debug(f"Using chunk size of {chunk_size} words for forced splits")
                
                for i in range(chunk_size - 1, len(words) - min_words_per_segment, chunk_size):
                    if i < len(words) - 1:  # Don't add the very last word as a split point
                        # Skip if this would separate a determiner from its noun
                        if words[i].lower().strip(',.;:!?') in determiners:
                            logger.debug(f"Skipping forced split at word {i+1} ('{words[i]}') because it's a determiner")
                            # Try to move the split point back one word
                            if i > 0:
                                logger.debug(f"Moving split back to {i} ('{words[i-1]}')")
                                split_points.append(i-1)
                            continue
                            
                        # Skip if next word is a determiner - we want to keep it with what follows
                        if i + 1 < len(words) and words[i + 1].lower().strip(',.;:!?') in determiners:
                            logger.debug(f"Skipping forced split at word {i+1} ('{words[i]}') because next word '{words[i+1]}' is a determiner")
                            # Try to move the split point back one word
                            if i > 0:
                                logger.debug(f"Moving split back to {i} ('{words[i-1]}')")
                                split_points.append(i-1)
                            continue
                            
                        logger.debug(f"Adding forced split at word {i+1} ('{words[i]}')")
                        split_points.append(i)
                
                # Sort split points to ensure they're in order
                split_points.sort()
                logger.debug(f"Forced split points: {split_points}")
            
            # Ensure the last segment has at least min_words_per_segment words
            if split_points and len(words) - split_points[-1] - 1 < min_words_per_segment:
                logger.debug(f"Last segment would be too short ({len(words) - split_points[-1] - 1} < {min_words_per_segment} words)")
                logger.debug(f"Removing last split point at word {split_points[-1]+1} ('{words[split_points[-1]]}')")
                # If the last segment would be too short, remove the last split point
                # This will merge the last few words with the previous segment
                split_points.pop()
            
            logger.debug(f"Final split points: {split_points}")
            
            # Process the text using found split points
            if not split_points:
                logger.debug("No valid split points found, keeping subtitle as is")
                # If still no split points (very short but still > max_length), 
                # just keep original
                new_subs.append(pysrt.SubRipItem(
                    index=counter,
                    start=sub.start,
                    end=sub.end,
                    text=sub.text.strip()
                ))
                counter += 1
                continue
                    
            # Calculate time per word for even distribution
            avg_word_duration = total_duration / len(words)
            logger.debug(f"Average word duration: {avg_word_duration} ms")
            current_start = sub.start.ordinal
            
            # Now create a subtitle for each segment
            last_index = 0
            for i, split_index in enumerate(split_points):
                # Extract text for this segment
                segment_words = words[last_index:split_index + 1]
                split_text = " ".join(segment_words).strip()
                
                # Calculate duration based on word count
                segment_duration = int(len(segment_words) * avg_word_duration)
                segment_end = current_start + segment_duration
                
                logger.debug(f"Creating segment {i+1}/{len(split_points)+1}: '{split_text}'")
                logger.debug(f"  Word range: {last_index+1}-{split_index+1} ({len(segment_words)} words)")
                logger.debug(f"  Time: {current_start} -> {segment_end} (duration: {segment_duration} ms)")
                
                # Create subtitle
                new_subs.append(pysrt.SubRipItem(
                    index=counter,
                    start=pysrt.SubRipTime.from_ordinal(current_start),
                    end=pysrt.SubRipTime.from_ordinal(segment_end),
                    text=split_text
                ))
                counter += 1
                
                # Update for next segment
                current_start = segment_end
                last_index = split_index + 1
            
            # Add final segment if there are remaining words
            if last_index < len(words):
                remaining_words = words[last_index:]
                remaining_text = " ".join(remaining_words).strip()
                
                logger.debug(f"Creating final segment: '{remaining_text}'")
                logger.debug(f"  Word range: {last_index+1}-{len(words)} ({len(remaining_words)} words)")
                logger.debug(f"  Time: {current_start} -> {sub.end.ordinal} (duration: {sub.end.ordinal - current_start} ms)")
                
                new_subs.append(pysrt.SubRipItem(
                    index=counter,
                    start=pysrt.SubRipTime.from_ordinal(current_start),
                    end=sub.end,
                    text=remaining_text
                ))
                counter += 1
        
        logger.info(f"Split long subtitles into {len(new_subs)} total subtitles")
        logger.debug(f"Saving to {output_srt}")
        new_subs.save(output_srt, encoding='utf-8')

    def calculate_text_similarity(self, text1, text2):
        """Calculate text similarity between two strings"""
        logger.debug(f"Calculating similarity between:")
        logger.debug(f"  Text 1: '{text1}'")
        logger.debug(f"  Text 2: '{text2}'")
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        logger.debug(f"  Text 1 words: {words1}")
        logger.debug(f"  Text 2 words: {words2}")
        
        # If either text is empty, return 0
        if not words1 or not words2:
            logger.debug("One or both texts are empty, similarity = 0")
            return 0
            
        # Calculate Jaccard similarity (intersection over union)
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        logger.debug(f"  Intersection ({len(intersection)} words): {intersection}")
        logger.debug(f"  Union ({len(union)} words): {union}")
        
        # Additional check for repeated single words
        if text1.lower() == text2.lower():
            logger.debug("Texts are identical, similarity = 1.0")
            return 1.0
        
        # Fix: Use len(union) for comparison instead of the set itself
        similarity = len(intersection) / len(union) if len(union) > 0 else 0
        logger.debug(f"  Similarity calculation: {len(intersection)}/{len(union)} = {similarity}")
        return similarity

    def clear_srt_files(self):
        """Clear all selected SRT files"""
        logger.debug("Clearing all selected SRT files")
        self.file_template_pairs = []
        self.processed_srt_paths = []
        self.update_files_display()
        self.update_ui_state()

    def on_tree_double_click(self, event):
        """Handle double-click on treeview to change template"""
        logger.debug("Tree double-click detected")
        
        # Get the item and column
        item = self.files_tree.selection()[0]
        column = self.files_tree.identify_column(event.x)
        
        logger.debug(f"Double-clicked on item: {item}, column: {column}")
        
        # Only allow template changes (column #2)
        if column == '#2':  # Template column
            self.change_template_for_file(item)
    
    def change_template_for_file(self, tree_item):
        """Show dialog to change template for a specific file"""
        logger.debug("Changing template for file")
        
        if not self.mediaPoolItemsList:
            logger.warning("No templates available")
            messagebox.showwarning("No Templates", "No Text+ templates found in Media Pool")
            return
        
        # Get the index of the selected item
        item_index = self.files_tree.index(tree_item)
        
        if item_index >= len(self.file_template_pairs):
            logger.error(f"Invalid item index: {item_index}")
            return
        
        current_file, current_template = self.file_template_pairs[item_index]
        logger.debug(f"Changing template for file: {os.path.basename(current_file)}")
        
        # Create template selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Select Template for {os.path.basename(current_file)}")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        tk.Label(dialog, text=f"Select template for:\n{os.path.basename(current_file)}", 
                font=("Arial", 10)).pack(pady=10)
        
        # Template listbox with scrollbar
        list_frame = tk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        template_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        template_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=template_listbox.yview)
        
        # Populate template list
        template_names = []
        current_selection = 0
        for i, template in enumerate(self.mediaPoolItemsList):
            name = template.GetClipProperty()['Clip Name']
            template_names.append(name)
            template_listbox.insert(tk.END, name)
            
            # Select current template
            if template == current_template:
                current_selection = i
        
        template_listbox.selection_set(current_selection)
        template_listbox.see(current_selection)
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def on_ok():
            selected_indices = template_listbox.curselection()
            if selected_indices:
                selected_index = selected_indices[0]
                new_template = self.mediaPoolItemsList[selected_index]
                logger.info(f"Changed template for {os.path.basename(current_file)} to {template_names[selected_index]}")
                
                # Update the file-template pair
                self.file_template_pairs[item_index] = (current_file, new_template)
                self.update_files_display()
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        tk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Allow double-click to select
        def on_listbox_double_click(event):
            on_ok()
        
        template_listbox.bind('<Double-1>', on_listbox_double_click)

    def update_files_display(self):
        """Update the treeview to show selected files and their templates"""
        logger.debug("Updating files display")
        
        # Clear existing items
        for item in self.files_tree.get_children():
            self.files_tree.delete(item)
        
        # Add current file-template pairs
        for i, (file_path, template) in enumerate(self.file_template_pairs):
            file_name = os.path.basename(file_path)
            template_name = template.GetClipProperty()['Clip Name'] if template else "No template"
            
            self.files_tree.insert('', 'end', values=(file_name, template_name))
            logger.debug(f"Added to display: {file_name} -> {template_name}")
        
        # Update file count label
        if len(self.file_template_pairs) == 0:
            self.file_count_label.config(text="No SRT files selected")
        elif len(self.file_template_pairs) == 1:
            self.file_count_label.config(text="1 SRT file selected")
        else:
            self.file_count_label.config(text=f"{len(self.file_template_pairs)} SRT files selected")

    def update_ui_state(self):
        """Update UI button states based on current selection"""
        logger.debug("Updating UI state")
        
        # Check if all files have templates assigned
        has_files = len(self.file_template_pairs) > 0
        all_have_templates = all(template is not None for _, template in self.file_template_pairs)
        
        # Enable add button if we have files and all have templates
        if has_files and all_have_templates:
            logger.debug("Enabling Add Subtitles button (all files have templates)")
            self.add_subs_button.config(state="normal")
            self.status_label.config(text=f"Ready to process {len(self.file_template_pairs)} file(s)")
        else:
            logger.debug("Disabling Add Subtitles button")
            self.add_subs_button.config(state="disabled")
            if not has_files:
                self.status_label.config(text="Please select SRT file(s)")
            elif not all_have_templates:
                files_without_templates = sum(1 for _, template in self.file_template_pairs if template is None)
                self.status_label.config(text=f"{files_without_templates} file(s) need template assignment")

    def process_subtitles(self):
        """Main function to process subtitles - handles both single and multiple files"""
        logger.info("Starting subtitle processing")
        
        if not self.file_template_pairs:
            logger.error("No SRT files selected")
            self.show_error("No SRT files selected")
            return
        
        # Check if all files have templates
        files_without_templates = [file_path for file_path, template in self.file_template_pairs if template is None]
        if files_without_templates:
            logger.error(f"{len(files_without_templates)} files without templates")
            self.show_error(f"{len(files_without_templates)} files need template assignment")
            return
        
        processing_mode = self.processing_mode_var.get()
        logger.info(f"Processing mode: {processing_mode}")
        
        if processing_mode == "sequential":
            self.process_files_sequentially()
        else:
            self.process_files_as_batch()

    def process_files_sequentially(self):
        """Process each SRT file one by one, placing each on separate track groups - FIXED track calculation"""
        logger.info(f"Processing {len(self.file_template_pairs)} files sequentially")
        
        # Show progress bar
        self.progress_bar.pack(fill=tk.X, padx=20, pady=5)
        self.progress_var.set(0)
        self.root.update()
        
        total_files = len(self.file_template_pairs)
        successful_files = 0
        failed_files = []
        
        try:
            # Get project and timeline once
            project_manager = resolve.GetProjectManager()
            project = project_manager.GetCurrentProject()
            if not project:
                logger.error("No project found")
                self.show_error("No project found")
                return
            
            timeline = project.GetCurrentTimeline()
            if not timeline:
                logger.error("No timeline found")
                self.show_error("No timeline found")
                return
            
            # Get frame rate once
            resolve.OpenPage("edit")
            frame_rate = int(timeline.GetSetting("timelineFrameRate"))
            if frame_rate == 29:
                frame_rate = 30
            logger.info(f"Using frame rate: {frame_rate}")
            
            # Get current track count to determine where to start placing subtitles
            initial_track_count = timeline.GetTrackCount("video")
            logger.info(f"Initial video track count: {initial_track_count}")
            
            # Track where the next file should start
            next_available_track = initial_track_count + 1  # Start after existing tracks
            
            for file_index, (srt_file_path, template) in enumerate(self.file_template_pairs):
                try:
                    logger.info(f"Processing file {file_index + 1}/{total_files}: {os.path.basename(srt_file_path)}")
                    logger.info(f"File {file_index + 1} will start at track {next_available_track}")
                    self.status_label.config(text=f"Processing file {file_index + 1}/{total_files}: {os.path.basename(srt_file_path)}")
                    self.root.update()
                    
                    # Set current file and template for processing
                    self.srt_file_path = srt_file_path
                    self.current_template = template
                    
                    # Preprocess this specific file
                    if self.should_preprocess():
                        logger.info(f"Preprocessing file: {os.path.basename(srt_file_path)}")
                        if not self.preprocess_srt():
                            logger.error(f"Preprocessing failed for file: {os.path.basename(srt_file_path)}")
                            failed_files.append(os.path.basename(srt_file_path))
                            continue
                        processed_path = self.processed_srt_path
                    else:
                        logger.debug(f"No preprocessing needed for file: {os.path.basename(srt_file_path)}")
                        processed_path = srt_file_path
                    
                    # Use current available track as starting point
                    target_track_start = next_available_track
                    
                    # Process file and count actual tracks used
                    highest_track_used = self.add_subtitles_for_file_and_count_tracks(
                        processed_path, frame_rate, timeline, project, target_track_start, template
                    )
                    
                    if highest_track_used > 0:
                        successful_files += 1
                        tracks_used = highest_track_used - target_track_start + 1
                        logger.info(f"Successfully processed file: {os.path.basename(srt_file_path)}")
                        logger.info(f"File {file_index + 1} used tracks {target_track_start} to {highest_track_used} ({tracks_used} total)")
                        
                        # Simply set next file to start after the highest track used
                        next_available_track = highest_track_used + 1
                        logger.info(f"Next file will start at track {next_available_track}")
                    else:
                        failed_files.append(os.path.basename(srt_file_path))
                        logger.error(f"Failed to process file: {os.path.basename(srt_file_path)}")
                    
                    # Update progress
                    progress = ((file_index + 1) / total_files) * 100
                    self.progress_var.set(progress)
                    self.root.update()
                    
                except Exception as e:
                    logger.error(f"Error processing file {os.path.basename(srt_file_path)}: {str(e)}", exc_info=True)
                    failed_files.append(os.path.basename(srt_file_path))
            
            # Save project
            logger.info("Saving project...")
            project_manager.SaveProject()
            
            # Hide progress bar
            self.progress_bar.pack_forget()
            
            # Show completion message
            if successful_files == total_files:
                message = f"Successfully processed all {total_files} SRT files!"
                self.status_label.config(text=f"Completed: {successful_files}/{total_files} files processed")
            else:
                message = f"Processed {successful_files}/{total_files} files successfully."
                if failed_files:
                    message += f"\nFailed files: {', '.join(failed_files)}"
                self.status_label.config(text=f"Completed with errors: {successful_files}/{total_files} files")
                messagebox.showwarning("Partial Success", message)
            
            logger.info(f"Sequential processing complete: {successful_files}/{total_files} successful")
            logger.info(f"Final track used: {next_available_track - 1}, Total tracks added: {next_available_track - initial_track_count - 1}")
            
        except Exception as e:
            logger.error(f"Error in sequential processing: {str(e)}", exc_info=True)
            self.progress_bar.pack_forget()
            self.show_error(f"Error processing files: {str(e)}")

    def add_subtitles_for_file_and_count_tracks(self, processed_srt_path, frame_rate, timeline, project, target_track_start, template):
        """
        Process file and return the highest track number actually used.
        Returns: Highest track number used (0 if failed)
        """
        logger.info(f"Adding subtitles from {os.path.basename(processed_srt_path)} starting at track {target_track_start}")
        template_name = template.GetClipProperty()['Clip Name']
        logger.info(f"Using template: {template_name}")
        
        try:
            # Get subtitle delay from UI
            try:
                subtitle_delay_seconds = float(self.delay_var.get())
                logger.info(f"Using subtitle delay: {subtitle_delay_seconds} seconds")
            except ValueError:
                logger.warning("Invalid delay value, using default 0.2 seconds")
                subtitle_delay_seconds = 0.2
            
            # Convert delay to frames
            delay_frames = int(subtitle_delay_seconds * frame_rate)
            logger.info(f"Subtitle delay: {subtitle_delay_seconds}s = {delay_frames} frames")
            
            media_pool = project.GetMediaPool()
            
            # Parse SRT file (same as before)
            subs = []
            try:
                logger.debug("Opening SRT file for parsing")
                with open(processed_srt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                logger.debug(f"Read {len(lines)} lines from SRT file")
                
                i = 0
                while i < len(lines):
                    try:
                        # Skip empty lines
                        while i < len(lines) and not lines[i].strip():
                            i += 1
                        
                        if i >= len(lines):
                            break
                            
                        # Find index line (a number)
                        if not lines[i].strip().isdigit():
                            logger.debug(f"Line {i+1} is not a subtitle index: {lines[i].strip()}")
                            i += 1
                            continue
                        
                        subtitle_index = lines[i].strip()
                        
                        # Get timestamp line
                        i += 1
                        if i >= len(lines):
                            logger.warning(f"Unexpected end of file after subtitle index {subtitle_index}")
                            break
                            
                        timestamp_line = lines[i].strip()
                        if " --> " not in timestamp_line:
                            logger.warning(f"Invalid timestamp format for subtitle {subtitle_index}: {timestamp_line}")
                            i += 1
                            continue
                        
                        start_time, end_time = timestamp_line.split(" --> ")
                        
                        # Get text (could be multiple lines)
                        i += 1
                        text_lines = []
                        while i < len(lines) and lines[i].strip():
                            text_lines.append(lines[i].strip())
                            i += 1
                        
                        text = " ".join(text_lines)
                        
                        # Convert to frames
                        start_frames = self.time_to_frames(start_time, frame_rate)
                        end_frames = self.time_to_frames(end_time, frame_rate)
                        
                        # Apply delay to both start and end times
                        start_frames_delayed = start_frames + delay_frames
                        end_frames_delayed = end_frames + delay_frames
                        
                        # Calculate position and duration
                        timeline_pos = timeline.GetStartFrame() + start_frames_delayed
                        duration = end_frames_delayed - start_frames_delayed
                        
                        # Ensure minimum duration
                        if duration < 1:
                            duration = 1
                            
                        subs.append((timeline_pos, duration, text))
                    except Exception as e:
                        logger.error(f"Error processing subtitle at line {i}: {str(e)}")
                        i += 1
                        
                if not subs:
                    logger.error("No valid subtitles found in SRT file")
                    return 0
                
                logger.info(f"Successfully parsed {len(subs)} subtitles with {subtitle_delay_seconds}s delay")
                    
            except Exception as e:
                logger.error(f"Error parsing SRT file: {str(e)}", exc_info=True)
                return 0
            
            # Assign subtitles to tracks
            logger.debug("Assigning subtitles to tracks")
            track_assignments, assigned_tracks = self.assign_tracks_to_subtitles(subs)
            
            # Simply get the actual highest track number that will be used
            unique_tracks = set(assigned_tracks)
            logger.info(f"File assigned subtitles to {len(unique_tracks)} tracks: {sorted(unique_tracks)}")
            
            # Prepare timeline tracks - ensure we have enough tracks
            current_track_count = timeline.GetTrackCount("video")
            logger.debug(f"Current video track count: {current_track_count}")
            
            # Convert relative tracks to actual timeline tracks
            final_assigned_tracks = []
            for track in assigned_tracks:
                final_track = target_track_start + track - 1
                final_assigned_tracks.append(final_track)
            
            # Find the highest actual track number we'll use
            highest_track = max(final_assigned_tracks)
            logger.info(f"File will use timeline tracks {min(final_assigned_tracks)} to {highest_track}")
            
            # Add tracks if needed
            tracks_to_add = highest_track - current_track_count
            if tracks_to_add > 0:
                logger.debug(f"Adding {tracks_to_add} tracks to accommodate track {highest_track}")
                for _ in range(tracks_to_add):
                    timeline.AddTrack("video")
            
            # Add subtitle clips to timeline using the specified template
            subtitle_clips = []
            for i, (timeline_pos, duration, text) in enumerate(subs):
                track_num = final_assigned_tracks[i]
                
                newClip = {
                    "mediaPoolItem": template,  # Use the specified template for this file
                    "startFrame": 0,
                    "endFrame": duration,
                    "trackIndex": track_num,
                    "recordFrame": timeline_pos
                }
                subtitle_clips.append(newClip)
            
            logger.info(f"Attempting to add {len(subtitle_clips)} clips to timeline using template '{template_name}'")
            success = media_pool.AppendToTimeline(subtitle_clips)
            
            if not success:
                logger.error("Failed to add clips to timeline")
                return 0
            
            logger.info("Successfully added clips to timeline")

            # Organize subtitles by track for text updating (same as before)
            subs_by_track = {}
            for idx, (timeline_pos, duration, text) in enumerate(subs):
                track_num = final_assigned_tracks[idx]
                if track_num not in subs_by_track:
                    subs_by_track[track_num] = []
                subs_by_track[track_num].append(text)

            # Initialize tracking for text similarity and size variation (same as before)
            prev_texts_by_track = {}
            current_pattern_index_by_track = {}
            prev_sizes_by_track = {}
            size_pattern = [1.4, 0.7]
            
            logger.info("Updating subtitle text...")
            for track_num, texts in sorted(subs_by_track.items()):
                try:
                    logger.debug(f"Processing track {track_num} with {len(texts)} subtitles")
                    
                    # Initialize for this track if not already done
                    if track_num not in prev_texts_by_track:
                        prev_texts_by_track[track_num] = []
                    if track_num not in current_pattern_index_by_track:
                        current_pattern_index_by_track[track_num] = 0
                    if track_num not in prev_sizes_by_track:
                        prev_sizes_by_track[track_num] = []
                        
                    sub_list = timeline.GetItemListInTrack('video', track_num)
                    if not sub_list:
                        logger.warning(f"No items found in track {track_num}")
                        continue
                        
                    logger.debug(f"Found {len(sub_list)} items in track {track_num}")
                    sub_list.sort(key=lambda clip: clip.GetStart())
                    
                    for i, clip in enumerate(sub_list):
                        if i < len(texts):
                            logger.debug(f"Processing clip {i+1}/{len(sub_list)} in track {track_num}")
                            clip.SetClipColor('Orange')
                            text = texts[i]
                            
                            # FIXED: Clean ALL track markers before displaying
                            display_text = text
                            is_track2_text = False
                            is_track3_text = False
                            
                            if text.startswith('[TRACK2]'):
                                display_text = text[8:]
                                is_track2_text = True
                                logger.debug(f"Cleaned [TRACK2] marker: '{text}' -> '{display_text}'")
                            elif text.startswith('[TRACK3]'):
                                display_text = text[8:]
                                is_track3_text = True
                                logger.debug(f"Cleaned [TRACK3] marker: '{text}' -> '{display_text}'")
                            
                            # Check for similarity with previous subtitles
                            is_similar = False
                            for j, prev_text in enumerate(prev_texts_by_track[track_num][-3:]):
                                similarity = self.calculate_text_similarity(display_text, prev_text)
                                if similarity >= 0.7:
                                    is_similar = True
                                    break
                            
                            # Add this text to previous texts for future comparisons
                            prev_texts_by_track[track_num].append(display_text)
                            
                            # Format text (same logic as before)
                            max_length = 18
                            max_size = 0.12
                            words = display_text.split()
                            current_line = ""
                            lines_formatted = []
                            
                            for word in words:
                                if len(current_line) + len(word) + 1 <= max_length:
                                    current_line += word + " "
                                else:
                                    lines_formatted.append(current_line.strip())
                                    current_line = word + " "
                            if current_line:
                                lines_formatted.append(current_line.strip())
                            
                            # Calculate text size (same logic as before)
                            char_count = len(display_text.replace(" ", ""))
                            starting_size = 0.08
                            size_increase = max(0, 6 - char_count) * 0.1
                            new_size = min(starting_size + size_increase, max_size)
                            
                            # Apply size variation for similar subtitles
                            if is_similar:
                                current_idx = current_pattern_index_by_track[track_num]
                                size_multiplier = size_pattern[current_idx]
                                new_size = new_size * size_multiplier
                                current_pattern_index_by_track[track_num] = (current_idx + 1) % len(size_pattern)
                            else:
                                current_pattern_index_by_track[track_num] = 0
                            
                            # Ensure size stays within bounds
                            new_size = max(min(new_size, 0.18), 0.05)
                            
                            # Store size for potential track 3 inheritance
                            prev_sizes_by_track[track_num].append(new_size)
                            
                            # Special handling for track 3 size inheritance
                            final_size = new_size
                            if is_track3_text:
                                target_track = track_num - 1
                                
                                if target_track in prev_sizes_by_track and prev_sizes_by_track[target_track]:
                                    inherited_size = prev_sizes_by_track[target_track][-1]
                                    final_size = inherited_size
                                    logger.info(f"Track 3 inheriting exact size {inherited_size} from track {target_track}")
                                else:
                                    final_size = max(0.08, min(new_size, 0.12))
                                    logger.info(f"Track 3 using fallback size {final_size}")
                            
                            # Update text in Fusion composition (same logic as before)
                            comp = clip.GetFusionCompByIndex(1)
                            if comp is not None:
                                tools = comp.GetToolList()
                                
                                for tool_id, tool in tools.items():
                                    tool_name = tool.GetAttrs()['TOOLS_Name']
                                    
                                    if tool_name == 'Template':
                                        comp.SetActiveTool(tool)
                                        
                                        # Track positioning relative to this file's start track
                                        track_relative_to_start = track_num - target_track_start
                                        
                                        if track_relative_to_start == 0:  # Track 1 (relative)
                                            logger.debug(f"Track {track_num}: Lower position (track 1 relative)")
                                            tool.SetInput('StyledText', display_text)
                                        elif track_relative_to_start == 1:  # Track 2 (relative)
                                            logger.debug(f"Track {track_num}: Upper position (track 2 relative)")
                                            tool.SetInput('StyledText', "\n\n" + display_text)
                                        elif track_relative_to_start == 2:  # Track 3 (relative)
                                            logger.debug(f"Track {track_num}: Right-aligned position (track 3 relative)")
                                            
                                            base_spacing = 15
                                            prev_text_length = 0
                                            if i > 0 and track_num - 1 in subs_by_track:
                                                prev_track_texts = subs_by_track[track_num - 1]
                                                if i < len(prev_track_texts):
                                                    prev_text_length = len(prev_track_texts[i])
                                            
                                            length_adjustment = max(0, prev_text_length - len(display_text))
                                            total_spacing = base_spacing + (length_adjustment // 2)
                                            total_spacing = max(5, min(total_spacing, 30))
                                            
                                            right_aligned_text = " " * total_spacing + display_text
                                            tool.SetInput('StyledText', right_aligned_text)
                                        else:
                                            # For any additional tracks, alternate between upper and lower
                                            is_upper_position = track_relative_to_start % 2 != 0
                                            if is_upper_position:
                                                logger.debug(f"Track {track_num}: Upper position (track {track_relative_to_start + 1} relative)")
                                                tool.SetInput('StyledText', "\n\n" + display_text)
                                            else:
                                                logger.debug(f"Track {track_num}: Lower position (track {track_relative_to_start + 1} relative)")
                                                tool.SetInput('StyledText', display_text)
                                            
                                        tool.SetInput('Size', final_size)
                            
                            clip.SetClipColor('Teal')
                except Exception as e:
                    logger.error(f"Error updating subtitles in track {track_num}: {str(e)}", exc_info=True)
            
            logger.info(f"Successfully processed file: {os.path.basename(processed_srt_path)} with template '{template_name}'")
            
            # Return the highest track number actually used
            return highest_track
            
        except Exception as e:
            logger.error(f"Error in add_subtitles_for_file_with_track_info: {str(e)}", exc_info=True)
            return 0

    def process_files_as_batch(self):
        """Process all files as one combined batch - NOTE: All files will use the first file's template"""
        logger.info(f"Processing {len(self.file_template_pairs)} files as one batch")
        
        # Show progress bar
        self.progress_bar.pack(fill=tk.X, padx=20, pady=5)
        self.progress_var.set(0)
        self.root.update()
        
        try:
            # For batch processing, use the template from the first file
            batch_template = self.file_template_pairs[0][1]
            logger.info(f"Using template '{batch_template.GetClipProperty()['Clip Name']}' for batch processing")
            
            # Preprocess all files first
            all_processed_paths = []
            total_files = len(self.file_template_pairs)
            
            for file_index, (srt_file_path, template) in enumerate(self.file_template_pairs):
                logger.info(f"Preprocessing file {file_index + 1}/{total_files}: {os.path.basename(srt_file_path)}")
                self.status_label.config(text=f"Preprocessing file {file_index + 1}/{total_files}: {os.path.basename(srt_file_path)}")
                self.root.update()
                
                # Set current file for processing
                self.srt_file_path = srt_file_path
                
                if self.should_preprocess():
                    if not self.preprocess_srt():
                        logger.error(f"Preprocessing failed for file: {os.path.basename(srt_file_path)}")
                        self.show_error(f"Preprocessing failed for file: {os.path.basename(srt_file_path)}")
                        return
                    all_processed_paths.append(self.processed_srt_path)
                else:
                    all_processed_paths.append(srt_file_path)
                
                # Update progress for preprocessing phase (first 50%)
                progress = ((file_index + 1) / total_files) * 50
                self.progress_var.set(progress)
                self.root.update()
            
            # Combine all processed SRT files into one
            logger.info("Combining all SRT files into one batch")
            self.status_label.config(text="Combining all files into one batch...")
            self.root.update()
            
            combined_srt_path = self.combine_srt_files(all_processed_paths)
            
            # Update progress to 75%
            self.progress_var.set(75)
            self.root.update()
            
            # Process the combined file using the batch template
            logger.info("Processing combined SRT file")
            self.status_label.config(text="Adding combined subtitles to timeline...")
            self.root.update()
            
            self.srt_file_path = combined_srt_path
            self.processed_srt_path = combined_srt_path
            self.template_text = batch_template  # Set the template for the original add_subtitles method
            
            # Call the original add_subtitles method
            self.add_subtitles()
            
            # Update progress to 100%
            self.progress_var.set(100)
            self.root.update()
            
            # Hide progress bar
            self.progress_bar.pack_forget()
            
            # Clean up combined file
            try:
                os.remove(combined_srt_path)
                logger.debug(f"Cleaned up combined SRT file: {combined_srt_path}")
            except:
                pass
            
            template_name = batch_template.GetClipProperty()['Clip Name']
            self.status_label.config(text=f"Batch processing complete: {total_files} files combined and processed with '{template_name}'")
            messagebox.showinfo("Success", f"Successfully processed all {total_files} SRT files as one batch using template '{template_name}'!")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
            self.progress_bar.pack_forget()
            self.show_error(f"Error processing files: {str(e)}")

    def combine_srt_files(self, srt_file_paths):
        """Combine multiple SRT files into one, adjusting timing to be sequential"""
        logger.info(f"Combining {len(srt_file_paths)} SRT files")
        
        combined_subs = pysrt.SubRipFile()
        counter = 1
        current_end_time = 0  # Track the end time of the last subtitle
        
        for file_index, srt_path in enumerate(srt_file_paths):
            logger.debug(f"Processing file {file_index + 1}: {os.path.basename(srt_path)}")
            
            try:
                subs = pysrt.open(srt_path, encoding='utf-8')
                logger.debug(f"Loaded {len(subs)} subtitles from {os.path.basename(srt_path)}")
                
                # Add a gap between files (2 seconds)
                gap_duration = 2000  # 2 seconds in milliseconds
                if file_index > 0:
                    current_end_time += gap_duration
                    logger.debug(f"Added {gap_duration}ms gap after previous file")
                
                for sub in subs:
                    # Calculate new timing based on current position
                    original_start = sub.start.ordinal
                    original_end = sub.end.ordinal
                    duration = original_end - original_start
                    
                    new_start = current_end_time + original_start
                    new_end = new_start + duration
                    
                    # Create new subtitle with adjusted timing
                    new_sub = pysrt.SubRipItem(
                        index=counter,
                        start=pysrt.SubRipTime.from_ordinal(new_start),
                        end=pysrt.SubRipTime.from_ordinal(new_end),
                        text=sub.text
                    )
                    combined_subs.append(new_sub)
                    counter += 1
                
                # Update current_end_time to the end of this file
                if subs:
                    last_sub = subs[-1]
                    file_end_time = current_end_time + last_sub.end.ordinal
                    current_end_time = file_end_time
                    logger.debug(f"File {file_index + 1} ends at {current_end_time}ms")
                
            except Exception as e:
                logger.error(f"Error processing file {os.path.basename(srt_path)}: {str(e)}")
                raise
        
        # Save combined file
        base_dir = os.path.dirname(srt_file_paths[0])
        combined_path = os.path.join(base_dir, "combined_batch.srt")
        combined_subs.save(combined_path, encoding='utf-8')
        
        logger.info(f"Combined {len(srt_file_paths)} files into {len(combined_subs)} total subtitles")
        logger.debug(f"Saved combined file to: {combined_path}")
        
        return combined_path

    def find_text_templates(self):
        """Find Text+ templates in the media pool"""
        logger.info("Searching for Text+ templates in media pool")
        try:
            project_manager = resolve.GetProjectManager()
            self.project = project_manager.GetCurrentProject()
            
            if not self.project:
                logger.error("No project found in Resolve")
                return
            
            logger.debug("Getting media pool and root folder")
            self.mediaPool = self.project.GetMediaPool()
            root_folder = self.mediaPool.GetRootFolder()
            
            # Clear previous list
            self.mediaPoolItemsList = []
            
            # Search for Text+ templates
            logger.debug("Starting recursive search for Text+ templates")
            self.recursive_search(root_folder)
            
            if not self.mediaPoolItemsList:
                logger.warning("No Text+ templates found in Media Pool")
            else:
                logger.info(f"Found {len(self.mediaPoolItemsList)} Text+ templates")
                template_names = [item.GetClipProperty()['Clip Name'] for item in self.mediaPoolItemsList]
                logger.debug(f"Template names: {template_names}")
        except Exception as e:
            logger.error(f"Error finding Text+ templates: {str(e)}", exc_info=True)

    def should_preprocess(self):
        """Check if any preprocessing options are selected"""
        return (self.split_long_var.get() or 
                self.split_punct_var.get() or 
                self.split_commas_var.get() or 
                self.add_commas_var.get() or 
                self.add_commas_after_var.get() or 
                self.linguistic_segment_var.get() or
                self.add_duplicate_counters_var.get())

    def add_punctuation_after_interjections(self, input_file, output_file):
        """
        Add punctuation (commas or periods) after interjections based on context and rules.
        This enhances natural speech flow by adding appropriate pauses after interjections.
        """
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Adding punctuation after interjections from {input_file} to {output_file}")
        
        # Define interjection punctuation rules
        punctuation_rules = {
            # Rule 0: Compound interjections (treat as single units)
            'compound_interjections': {
                'enabled': True,
                'phrases': [
                    # Religious/exclamatory compounds
                    'jesus christ', 'holy shit', 'holy fuck', 'oh my god', 'my god', 'good god',
                    'god damn', 'goddamn', 'jesus fucking christ', 'for fucks sake', 'for fuck sake',
                    # Mild compound interjections
                    'oh well', 'oh yeah', 'oh no', 'oh boy', 'ah yes', 'ah well', 'well then',
                    'okay then', 'alright then', 'you know what', 'let me see', 'let me think',
                    # Expressive compounds
                    'holy cow', 'holy crap', 'what the hell', 'what the fuck', 'son of a bitch'
                ],
                'punctuation': {
                    # Strong compound interjections get commas (they lead into statements)
                    'jesus christ': ',', 'holy shit': ',', 'holy fuck': ',', 'oh my god': ',',
                    'my god': ',', 'good god': ',', 'god damn': ',', 'goddamn': ',',
                    'jesus fucking christ': ',', 'for fucks sake': ',', 'for fuck sake': ',',
                    'holy cow': ',', 'holy crap': ',', 'what the hell': ',', 'what the fuck': ',',
                    'son of a bitch': ',',
                    # Mild compounds get commas
                    'oh well': ',', 'oh yeah': ',', 'oh no': ',', 'oh boy': ',',
                    'ah yes': ',', 'ah well': ',', 'well then': ',', 'okay then': ',',
                    'alright then': ',', 'you know what': ',', 'let me see': ',', 'let me think': ','
                },
                'description': 'Handle compound interjections as single units'
            },
            
            # Rule 1: Strong interjections that usually get periods (complete thoughts)
            'period_interjections': {
                'enabled': getattr(self, 'period_interjections_var', tk.BooleanVar(value=True)).get(),
                'words': ['damn', 'fuck', 'shit', 'wow', 'whoa', 'ouch', 
                        'yikes', 'oops', 'ugh', 'oof', 'phew', 'boo', 'yay'],
                'description': 'Add periods after strong emotional interjections'
            },
            
            # Rule 2: Mild interjections that usually get commas (lead into statements)
            'comma_interjections': {
                'enabled': getattr(self, 'comma_interjections_var', tk.BooleanVar(value=True)).get(),
                'words': ['oh', 'ah', 'well', 'yeah', 'okay', 'right', 'now', 'so', 'actually', 
                        'basically', 'honestly', 'seriously', 'look', 'listen', 'see', 'hey',
                        'um', 'uh', 'er', 'hmm', 'mhm', 'eh'],
                'description': 'Add commas after mild interjections that lead into statements'
            },
            
            # Rule 3: Context-based decisions
            'context_based': {
                'enabled': getattr(self, 'context_based_var', tk.BooleanVar(value=True)).get(),
                'min_words_after': 2,  # Minimum words after interjection to add punctuation
                'prefer_comma_before_pronouns': True,  # "oh, I think" vs "oh. I think"
                'prefer_period_before_names': True,    # "oh. Jesus" vs "oh, Jesus"
                'prefer_comma_before_verbs': True,     # "well, let's go" vs "well. let's go"
                'description': 'Use context to decide between comma and period'
            },
            
            # Rule 4: Sentence position rules
            'position_based': {
                'enabled': getattr(self, 'position_based_var', tk.BooleanVar(value=True)).get(),
                'start_of_sentence': 'comma',  # What to add at start: 'comma', 'period', 'auto'
                'mid_sentence': 'comma',       # What to add in middle: 'comma', 'period', 'auto'
                'description': 'Different punctuation based on position in sentence'
            }
        }
        
        try:
            # Parse SRT file
            subtitles = self._parse_srt_file(input_file)
            logger.debug(f"Loaded {len(subtitles)} subtitles for interjection punctuation")
            
            # Process each subtitle
            for i, subtitle in enumerate(subtitles):
                content = " ".join(subtitle["content"])
                original_content = content
                
                logger.debug(f"========== Processing subtitle {i+1}/{len(subtitles)} ==========")
                logger.debug(f"Original: '{content}'")
                
                # Find interjections that need punctuation
                modified_text = self._add_interjection_punctuation(content, punctuation_rules)
                
                if modified_text != content:
                    logger.info(f"Added punctuation after interjections:")
                    logger.info(f"  BEFORE: '{original_content}'")
                    logger.info(f"  AFTER:  '{modified_text}'")
                    subtitle["content"] = [modified_text]
                else:
                    logger.debug("No punctuation added")
            
            # Write the modified SRT file
            self._write_srt_file(subtitles, output_file)
            logger.info(f"Successfully processed {len(subtitles)} subtitles, adding punctuation after interjections")
            return True
            
        except Exception as e:
            logger.error(f"Error adding punctuation after interjections: {str(e)}")
            logger.exception(e)
            return False

    def _add_interjection_punctuation(self, content, rules):
        """
        Analyze content and add appropriate punctuation after interjections.
        Returns modified content with added punctuation.
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        if len(content.split()) < 2:  # Need at least 2 words to add punctuation
            return content
        
        logger.debug(f"Analyzing content for interjection punctuation: '{content}'")
        
        # Step 1: Handle compound interjections first
        content = self._handle_compound_interjections(content, rules)
        logger.debug(f"After compound interjections: '{content}'")
        
        # Step 2: Handle individual interjections
        words = content.split()
        logger.debug(f"Words for individual processing: {words}")
        
        # Get all interjection words
        period_words = rules['period_interjections']['words'] if rules['period_interjections']['enabled'] else []
        comma_words = rules['comma_interjections']['words'] if rules['comma_interjections']['enabled'] else []
        all_interjections = set(period_words + comma_words)
        
        modified_words = []
        
        for i, word in enumerate(words):
            # Clean word for comparison (remove existing punctuation)
            clean_word = word.lower().strip('.,!?;:')
            
            logger.debug(f"--- Analyzing word {i+1}: '{word}' (clean: '{clean_word}') ---")
            
            # Check if this is an interjection
            if clean_word in all_interjections:
                logger.debug(f"Found interjection: '{clean_word}'")
                
                # Check if it already has punctuation
                if word.endswith(('.', ',', '!', '?', ';', ':')):
                    logger.debug(f"Interjection already has punctuation: '{word}'")
                    modified_words.append(word)
                    continue
                
                # Check if there are enough words after this interjection
                words_after = len(words) - i - 1
                min_words_after = rules['context_based']['min_words_after'] if rules['context_based']['enabled'] else 1
                
                if words_after < min_words_after:
                    logger.debug(f"Not enough words after interjection ({words_after} < {min_words_after})")
                    modified_words.append(word)
                    continue
                
                # Determine what punctuation to add
                punctuation_to_add = self._determine_interjection_punctuation(
                    clean_word, words, i, rules
                )
                
                if punctuation_to_add:
                    new_word = word + punctuation_to_add
                    logger.debug(f"Adding '{punctuation_to_add}' after '{word}' → '{new_word}'")
                    modified_words.append(new_word)
                else:
                    logger.debug(f"No punctuation added to '{word}'")
                    modified_words.append(word)
            else:
                modified_words.append(word)
        
        result = ' '.join(modified_words)
        logger.debug(f"Final result: '{result}'")
        return result

    def _handle_compound_interjections(self, content, rules):
        """
        Handle compound interjections (multi-word phrases) before processing individual words.
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        if not rules['compound_interjections']['enabled']:
            return content
        
        compound_phrases = rules['compound_interjections']['phrases']
        punctuation_map = rules['compound_interjections']['punctuation']
        
        # Sort by length (longest first) to avoid partial matches
        sorted_phrases = sorted(compound_phrases, key=len, reverse=True)
        
        modified_content = content.lower()  # Work with lowercase for matching
        original_content = content  # Keep original for case preservation
        
        for phrase in sorted_phrases:
            logger.debug(f"Checking for compound interjection: '{phrase}'")
            
            # Check if phrase exists at the start of the content
            if modified_content.startswith(phrase):
                # Check if it's followed by a space and more content
                if len(modified_content) > len(phrase) and modified_content[len(phrase)].isspace():
                    # Check if it already has punctuation
                    phrase_end_in_original = len(phrase)
                    if phrase_end_in_original < len(original_content):
                        next_char = original_content[phrase_end_in_original]
                        if next_char in '.,!?;:':
                            logger.debug(f"Compound interjection '{phrase}' already has punctuation")
                            continue
                    
                    # Get the appropriate punctuation
                    punctuation = punctuation_map.get(phrase, ',')
                    
                    # Preserve original case and add punctuation
                    original_phrase = original_content[:phrase_end_in_original]
                    rest_of_content = original_content[phrase_end_in_original:]
                    
                    new_content = original_phrase + punctuation + rest_of_content
                    
                    logger.info(f"Added punctuation to compound interjection:")
                    logger.info(f"  BEFORE: '{original_content}'")
                    logger.info(f"  AFTER:  '{new_content}'")
                    
                    return new_content
        
        return content

    def _determine_interjection_punctuation(self, interjection, words, position, rules):
        """
        Determine what punctuation to add after an interjection based on context and rules.
        Returns punctuation character or None.
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        logger.debug(f"Determining punctuation for '{interjection}' at position {position}")
        
        # Get the next word for context analysis
        next_word = words[position + 1] if position + 1 < len(words) else ""
        next_word_clean = next_word.lower().strip('.,!?;:')
        
        logger.debug(f"Next word: '{next_word}' (clean: '{next_word_clean}')")
        
        # Rule 1: Strong interjections usually get periods
        if (rules['period_interjections']['enabled'] and 
            interjection in rules['period_interjections']['words']):
            logger.debug(f"Rule 1: '{interjection}' is a strong interjection → period")
            return '.'
        
        # Rule 2: Mild interjections usually get commas
        if (rules['comma_interjections']['enabled'] and 
            interjection in rules['comma_interjections']['words']):
            
            # But use context to override if enabled
            if rules['context_based']['enabled']:
                # Prefer period before names/interjections
                if (rules['context_based']['prefer_period_before_names'] and 
                    next_word_clean in ['jesus', 'christ', 'god', 'damn', 'fuck', 'shit']):
                    logger.debug(f"Rule 2 + Context: '{interjection}' before name/strong word → period")
                    return '.'
                
                # Prefer comma before pronouns
                if (rules['context_based']['prefer_comma_before_pronouns'] and 
                    next_word_clean in ['i', 'you', 'we', 'they', 'he', 'she', 'it']):
                    logger.debug(f"Rule 2 + Context: '{interjection}' before pronoun → comma")
                    return ','
                
                # Prefer comma before verbs
                if (rules['context_based']['prefer_comma_before_verbs'] and 
                    self._is_likely_verb(next_word_clean)):
                    logger.debug(f"Rule 2 + Context: '{interjection}' before verb → comma")
                    return ','
            
            logger.debug(f"Rule 2: '{interjection}' is a mild interjection → comma")
            return ','
        
        # Rule 3: Position-based defaults
        if rules['position_based']['enabled']:
            if position == 0:  # Start of sentence
                default_punct = rules['position_based']['start_of_sentence']
            else:  # Mid sentence
                default_punct = rules['position_based']['mid_sentence']
            
            if default_punct == 'auto':
                # Auto-decide based on context
                if next_word_clean in ['i', 'you', 'we', 'they', 'he', 'she', 'it']:
                    return ','
                else:
                    return '.'
            elif default_punct in [',', '.']:
                logger.debug(f"Rule 3: Position-based → {default_punct}")
                return default_punct
        
        logger.debug(f"No punctuation determined for '{interjection}'")
        return None

    def _is_likely_verb(self, word):
        """
        Simple heuristic to check if a word is likely a verb.
        This is a basic implementation - could be enhanced with NLP.
        """
        common_verbs = [
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'shall',
            'go', 'goes', 'went', 'gone', 'going',
            'get', 'gets', 'got', 'gotten', 'getting',
            'make', 'makes', 'made', 'making',
            'take', 'takes', 'took', 'taken', 'taking',
            'come', 'comes', 'came', 'coming',
            'see', 'sees', 'saw', 'seen', 'seeing',
            'know', 'knows', 'knew', 'known', 'knowing',
            'think', 'thinks', 'thought', 'thinking',
            'want', 'wants', 'wanted', 'wanting',
            'need', 'needs', 'needed', 'needing',
            'look', 'looks', 'looked', 'looking',
            'find', 'finds', 'found', 'finding',
            'give', 'gives', 'gave', 'given', 'giving',
            'tell', 'tells', 'told', 'telling',
            'work', 'works', 'worked', 'working',
            'call', 'calls', 'called', 'calling',
            'try', 'tries', 'tried', 'trying',
            'ask', 'asks', 'asked', 'asking',
            'turn', 'turns', 'turned', 'turning',
            'move', 'moves', 'moved', 'moving',
            'play', 'plays', 'played', 'playing',
            'run', 'runs', 'ran', 'running',
            'hold', 'holds', 'held', 'holding',
            'bring', 'brings', 'brought', 'bringing',
            'happen', 'happens', 'happened', 'happening',
            'write', 'writes', 'wrote', 'written', 'writing',
            'sit', 'sits', 'sat', 'sitting',
            'stand', 'stands', 'stood', 'standing',
            'lose', 'loses', 'lost', 'losing',
            'pay', 'pays', 'paid', 'paying',
            'meet', 'meets', 'met', 'meeting',
            'include', 'includes', 'included', 'including',
            'continue', 'continues', 'continued', 'continuing',
            'set', 'sets', 'setting',
            'learn', 'learns', 'learned', 'learning',
            'change', 'changes', 'changed', 'changing',
            'lead', 'leads', 'led', 'leading',
            'understand', 'understands', 'understood', 'understanding',
            'watch', 'watches', 'watched', 'watching',
            'follow', 'follows', 'followed', 'following',
            'stop', 'stops', 'stopped', 'stopping',
            'create', 'creates', 'created', 'creating',
            'speak', 'speaks', 'spoke', 'spoken', 'speaking',
            'read', 'reads', 'reading',
            'allow', 'allows', 'allowed', 'allowing',
            'add', 'adds', 'added', 'adding',
            'spend', 'spends', 'spent', 'spending',
            'grow', 'grows', 'grew', 'grown', 'growing',
            'open', 'opens', 'opened', 'opening',
            'walk', 'walks', 'walked', 'walking',
            'win', 'wins', 'won', 'winning',
            'offer', 'offers', 'offered', 'offering',
            'remember', 'remembers', 'remembered', 'remembering',
            'love', 'loves', 'loved', 'loving',
            'consider', 'considers', 'considered', 'considering',
            'appear', 'appears', 'appeared', 'appearing',
            'buy', 'buys', 'bought', 'buying',
            'wait', 'waits', 'waited', 'waiting',
            'serve', 'serves', 'served', 'serving',
            'die', 'dies', 'died', 'dying',
            'send', 'sends', 'sent', 'sending',
            'build', 'builds', 'built', 'building',
            'stay', 'stays', 'stayed', 'staying',
            'fall', 'falls', 'fell', 'fallen', 'falling',
            'cut', 'cuts', 'cutting',
            'reach', 'reaches', 'reached', 'reaching',
            'kill', 'kills', 'killed', 'killing',
            'remain', 'remains', 'remained', 'remaining'
        ]
        
        return word.lower() in common_verbs

    def replace_commas_with_periods(self, input_file, output_file):
        """
        Smart comma-to-period replacement that considers interjections and overlap patterns.
        Converts commas to periods when they precede short interjections that should overlap with following text.
        """
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Smart comma-to-period replacement from {input_file} to {output_file}")
        
        # Define short interjections that typically overlap with following text
        short_interjections = [
            # Original interjections that should overlap
            'oh', 'ah', 'uh', 'um', 'well', 'yeah', 'yes', 'no', 'hey', 'hi', 'wow',
            'ouch', 'oops', 'yikes', 'damn', 'shit', 'fuck', 'jesus', 'god', 'christ',
            'phew', 'whoa', 'huh', 'hmm', 'ugh', 'oof', 'yay', 'boo', 'aw', 'eek',
            'gosh', 'geez', 'dang', 'darn', 'alright', 'okay', 'right', 'now', 'so',
            'but', 'and', 'or', 'then', 'yet', 'plus', 'also', 'too', 'either', 'neither',
            
            # Subject pronouns that start independent clauses (should become periods)
            'i', 'you', 'he', 'she', 'we', 'they', 'it',
            
            # Common independent clause starters (should become periods)
            'that', 'this', 'there', 'here',
            
            # Demonstrative pronouns starting new thoughts
            'those', 'these',
            
            # Common sentence starters that indicate new independent clauses
            'maybe', 'perhaps', 'probably', 'definitely', 'certainly', 'obviously',
            'actually', 'basically', 'honestly', 'seriously', 'really',
            
            # Time and sequence indicators that often start new clauses
            'later', 'after', 'before', 'during', 'while', 'when', 'since',
            
            # Modal auxiliaries that often start independent clauses
            'can', 'could', 'will', 'would', 'should', 'might', 'may', 'must',
            
            # Negative contractions that start clauses
            "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
            "shouldn't", "isn't", "aren't", "wasn't", "weren't"
        ]
        
        try:
            # Parse SRT file
            subtitles = self._parse_srt_file(input_file)
            logger.debug(f"Loaded {len(subtitles)} subtitles for smart comma replacement")
            
            for i, subtitle in enumerate(subtitles):
                content = " ".join(subtitle["content"])
                original_content = content
                
                logger.debug(f"Processing subtitle {i+1}/{len(subtitles)}: '{content}'")
                
                # Find all commas and analyze context
                comma_positions = []
                for pos, char in enumerate(content):
                    if char == ',':
                        comma_positions.append(pos)
                
                if len(comma_positions) == 0:
                    logger.debug("No commas found, skipping")
                    continue
                    
                logger.debug(f"Found {len(comma_positions)} comma(s) at positions: {comma_positions}")
                
                # Analyze each comma to decide if it should become a period
                modified_content = list(content)  # Convert to list for easy modification
                
                for comma_pos in comma_positions:
                    # Get text after this comma
                    after_comma = content[comma_pos + 1:].strip()
                    logger.debug(f"Analyzing comma at position {comma_pos}, text after: '{after_comma}'")
                    
                    if not after_comma:
                        continue
                    
                    # Split the text after comma into words
                    words_after = after_comma.split()
                    if not words_after:
                        continue
                    
                    first_word = words_after[0].lower().strip('.,!?;:')
                    logger.debug(f"First word after comma: '{first_word}'")
                    
                    # Check if first word is a short interjection
                    if first_word in short_interjections:
                        logger.debug(f"'{first_word}' is a short interjection")
                        
                        # Determine if this interjection should overlap with following text
                        should_convert = False
                        
                        # Rule 1: Single word interjections should usually overlap
                        if len(words_after) == 1:
                            should_convert = True
                            logger.debug("Single word interjection - should overlap")
                        
                        # Rule 2: Interjection + 1-3 more words should overlap
                        elif len(words_after) <= 4:
                            should_convert = True
                            logger.debug(f"Short phrase ({len(words_after)} words) - should overlap")
                        
                        # Rule 3: Common overlapping patterns
                        elif len(words_after) >= 2:
                            second_word = words_after[1].lower().strip('.,!?;:')
                            
                            # Patterns like "oh my god", "well then", "yeah right", etc.
                            overlapping_patterns = [
                                ('oh', ['my', 'god', 'no', 'yes', 'well', 'really', 'come', 'man']),
                                ('well', ['then', 'okay', 'now', 'yeah', 'sure', 'i', 'that', 'this']),
                                ('yeah', ['right', 'sure', 'okay', 'well', 'but', 'and', 'i', 'that']),
                                ('but', ['wait', 'no', 'yes', 'still', 'then', 'now', 'i', 'you']),
                                ('and', ['then', 'now', 'so', 'yet', 'still', 'also', 'i', 'you']),
                                ('so', ['then', 'now', 'what', 'yeah', 'i', 'you', 'we', 'they'])
                            ]
                            
                            for interjection, following_words in overlapping_patterns:
                                if first_word == interjection and second_word in following_words:
                                    should_convert = True
                                    logger.debug(f"Found overlapping pattern: '{first_word} {second_word}'")
                                    break
                        
                        if should_convert:
                            logger.info(f"Converting comma to period before interjection '{first_word}'")
                            logger.info(f"  Context: '...{content[max(0, comma_pos-10):comma_pos]}|,|{after_comma[:20]}...'")
                            modified_content[comma_pos] = '.'
                    
                    else:
                        logger.debug(f"'{first_word}' is not a short interjection, keeping comma")
                
                # Convert back to string and update if changed
                final_content = ''.join(modified_content)
                
                if final_content != content:
                    logger.info(f"Smart comma replacement made changes:")
                    logger.info(f"  BEFORE: '{original_content}'")
                    logger.info(f"  AFTER:  '{final_content}'")
                    subtitle["content"] = [final_content]
                else:
                    logger.debug("No changes made by smart comma replacement")
            
            # Write the modified SRT file
            self._write_srt_file(subtitles, output_file)
            logger.info(f"Successfully processed {len(subtitles)} subtitles with smart comma replacement")
            return True
            
        except Exception as e:
            logger.error(f"Error in smart comma replacement: {str(e)}")
            logger.exception(e)
            return False

    def fix_small_trailing_subtitles(self, input_file, output_file):
        """
        Post-process subtitles to move small trailing words to a 3rd track
        while extending the previous subtitle's duration. Only affects cases where
        a small singular word (1-2 words, short duration) follows an overlapping pair.
        """
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Fixing small trailing subtitles from {input_file} to {output_file}")
        
        try:
            import pysrt
            
            # Load SRT file
            subs = pysrt.open(input_file, encoding='utf-8')
            logger.debug(f"Loaded {len(subs)} subtitles for small trailing fix")
            
            # Get original subtitle groupings from comma splitting
            original_groups = getattr(self, '_original_subtitle_groups', [])
            logger.debug(f"Original subtitle groups: {original_groups}")
            
            # Create a mapping of subtitle index to group index
            subtitle_to_group = {}
            for group_idx, group in enumerate(original_groups):
                for subtitle_idx in group:
                    subtitle_to_group[subtitle_idx] = group_idx
            
            modifications_made = 0
            
            # Look for small trailing subtitles that should be moved to track 3
            for i in range(len(subs) - 1):
                current_sub = subs[i]
                next_sub = subs[i + 1]
                
                # Check if this is a candidate for 3rd track processing
                if self._is_small_trailing_candidate(current_sub, next_sub, subtitle_to_group, i):
                    
                    # Find the previous subtitle (should be the overlapping pair partner)
                    prev_sub = subs[i - 1] if i > 0 else None
                    
                    if prev_sub and self._should_extend_previous_subtitle(prev_sub, current_sub, subtitle_to_group, i):
                        
                        logger.info(f"Moving small trailing subtitle to 3rd track:")
                        logger.info(f"  Previous: '{prev_sub.text}' (will be extended)")
                        logger.info(f"  Current: '{current_sub.text}' (moving to track 3)")
                        logger.info(f"  Original timing - Prev ends: {prev_sub.end}, Current: {current_sub.start}-{current_sub.end}")
                        
                        # Extend previous subtitle to current subtitle's end time
                        original_prev_end = prev_sub.end.ordinal
                        prev_sub.end = current_sub.end
                        
                        # Mark current subtitle for track 3 by adding a special marker
                        # We'll use this marker later in track assignment
                        current_sub.text = f"[TRACK3]{current_sub.text}"
                        
                        logger.info(f"  Modified timing - Prev extended to: {prev_sub.end}")
                        logger.info(f"  Current marked for track 3: '{current_sub.text}'")
                        
                        modifications_made += 1
            
            # Save the modified SRT file
            subs.save(output_file, encoding='utf-8')
            
            logger.info(f"Fixed {modifications_made} small trailing subtitles for 3rd track placement")
            return True
            
        except Exception as e:
            logger.error(f"Error fixing small trailing subtitles: {str(e)}")
            logger.exception(e)
            return False

    def _is_small_trailing_candidate(self, current_sub, next_sub, subtitle_to_group, current_index):
        """
        Check if current subtitle is a small trailing word that should be moved to track 3.
        
        Criteria:
        1. Very short text (1-2 words)
        2. Short duration (less than 500ms)
        3. Simple words (common small words like "no", "yes", "ok", etc.)
        4. Part of a group with at least 2 other subtitles (indicating it came from comma splitting)
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        # Get basic properties
        text = current_sub.text.strip()
        words = text.split()
        duration_ms = current_sub.end.ordinal - current_sub.start.ordinal
        
        logger.debug(f"Analyzing subtitle {current_index + 1}: '{text}' ({len(words)} words, {duration_ms}ms)")
        
        # Must be 1-2 words
        if len(words) < 1 or len(words) > 2:
            logger.debug(f"  Not candidate: {len(words)} words (need 1-2)")
            return False
        
        # Must be short duration (less than 500ms)
        if duration_ms >= 500:
            logger.debug(f"  Not candidate: {duration_ms}ms duration (need <500ms)")
            return False
        
        # Must be a simple/common small word
        small_trailing_words = [
            'no', 'no.', 'yes', 'yes.', 'ok', 'ok.', 'okay', 'okay.',
            'yeah', 'yeah.', 'nah', 'nah.', 'wait', 'wait.', 'stop', 'stop.',
            'go', 'go.', 'now', 'now.', 'then', 'then.', 'so', 'so.',
            'but', 'but.', 'and', 'and.', 'or', 'or.', 'yet', 'yet.',
            'right', 'right.', 'wrong', 'wrong.', 'good', 'good.',
            'bad', 'bad.', 'fine', 'fine.', 'done', 'done.'
        ]
        
        first_word = words[0].lower().strip('.,!?;:')
        full_text = text.lower().strip()
        
        if not (first_word in small_trailing_words or full_text in small_trailing_words):
            logger.debug(f"  Not candidate: '{text}' not in small trailing words list")
            return False
        
        # Check if this subtitle is part of a group (came from comma splitting)
        current_group = subtitle_to_group.get(current_index)
        if current_group is None:
            logger.debug(f"  Not candidate: not part of any group")
            return False
        
        # Find how many subtitles are in this group
        group_size = sum(1 for idx, group in subtitle_to_group.items() if group == current_group)
        if group_size < 3:  # Need at least 3 subtitles in group for this to apply
            logger.debug(f"  Not candidate: group size {group_size} (need ≥3)")
            return False
        
        logger.debug(f"  CANDIDATE: '{text}' qualifies for 3rd track (group {current_group}, size {group_size})")
        return True

    def _should_extend_previous_subtitle(self, prev_sub, current_sub, subtitle_to_group, current_index):
        """
        Check if the previous subtitle should be extended to cover the current subtitle's duration.
        
        Criteria:
        1. Previous subtitle is in the same group (same original comma-split segment)
        2. Previous subtitle has reasonable length (not too short itself)
        3. The timing makes sense (previous ends before or around current start)
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        prev_index = current_index - 1
        prev_group = subtitle_to_group.get(prev_index)
        current_group = subtitle_to_group.get(current_index)
        
        # Must be in same group
        if prev_group != current_group:
            logger.debug(f"  Cannot extend: different groups (prev: {prev_group}, current: {current_group})")
            return False
        
        # Previous subtitle should not be too short itself
        prev_words = len(prev_sub.text.split())
        if prev_words < 1:
            logger.debug(f"  Cannot extend: previous subtitle too short ({prev_words} words)")
            return False
        
        # Check timing - previous should end before or shortly after current starts
        prev_end = prev_sub.end.ordinal
        current_start = current_sub.start.ordinal
        gap = current_start - prev_end
        
        if gap > 1000:  # Don't extend if gap is more than 1 second
            logger.debug(f"  Cannot extend: gap too large ({gap}ms)")
            return False
        
        logger.debug(f"  CAN EXTEND: previous subtitle '{prev_sub.text}' can be extended (gap: {gap}ms)")
        return True

    def _assign_tracks_original_logic(self, subs):
        """
        Original track assignment logic (moved here to preserve when handling track 3).
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        # Sort subtitles by start time
        sorted_indices = sorted(range(len(subs)), key=lambda i: subs[i][0])
        track_assignments = {1: [], 2: []}
        assigned_tracks = [None] * len(subs)
        
        # Keep track of the end time of the last subtitle on each track
        last_end_time = {1: 0, 2: 0}
        active_subtitle_on_track = {1: None, 2: None}
        
        for idx in sorted_indices:
            start, duration, text = subs[idx]
            end = start + duration
            logger.debug(f"Processing subtitle {idx}: Start={start}, End={end}, Text={text[:20]}...")
            
            # Finding the track with the earliest end time
            min_end_track = 1 if last_end_time[1] <= last_end_time[2] else 2
            
            # Default to track 1 if both tracks are free
            track_to_use = 1 if last_end_time[1] <= start and last_end_time[2] <= start else None
            
            if track_to_use is None:
                # If there's overlap, first check if there's a track that's completely free
                if start >= last_end_time[1]:
                    track_to_use = 1
                    logger.debug(f"  Track 1 is free, using it")
                elif start >= last_end_time[2]:
                    track_to_use = 2
                    logger.debug(f"  Track 2 is free, using it")
                else:
                    # Both tracks have active subtitles, find the one ending soonest
                    track_to_use = min_end_track
                    logger.debug(f"  Both tracks have active subtitles, using track {track_to_use} (ends soonest)")
                    
                    # Critical fix: If we're placing on track with an active subtitle,
                    # extend the virtual end time of the OTHER track to prevent a third subtitle
                    # from appearing before this pair is done
                    other_track = 3 - track_to_use  # If track_to_use is 1, other_track is 2, and vice versa
                    last_end_time[other_track] = max(last_end_time[other_track], end)
                    logger.debug(f"  Extended virtual end time of track {other_track} to {end} to prevent triple overlaps")
            
            # Assign to chosen track
            track_assignments[track_to_use].append(idx)
            assigned_tracks[idx] = track_to_use
            
            # Update the active subtitle and end time for the assigned track
            active_subtitle_on_track[track_to_use] = idx
            last_end_time[track_to_use] = end
            
            logger.debug(f"  Assigned subtitle {idx} to track {track_to_use} (ends at {end})")
        
        # Count usage of each track
        track_1_count = len(track_assignments[1])
        track_2_count = len(track_assignments[2])
        logger.info(f"Subtitles assigned to 2 tracks: Track 1: {track_1_count}, Track 2: {track_2_count}")
        
        return track_assignments, assigned_tracks

    def create_single_subtitle_overlaps(self, input_file, output_file):
        """
        Create overlaps for single short subtitles that should overlap with the next segment.
        FIXED VERSION: Prevents isolation of single pronouns and ensures minimum durations with proper gap limits.
        """
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Creating single subtitle overlaps from {input_file} to {output_file}")
        
        try:
            import pysrt
            
            # Load SRT file
            subs = pysrt.open(input_file, encoding='utf-8')
            logger.debug(f"Loaded {len(subs)} subtitles for single subtitle overlap processing")
            
            # Define overlap candidates
            overlap_candidates = [
                'but', 'and', 'or', 'so', 'well', 'oh', 'ah', 'yeah', 'no', 'yes',
                'now', 'then', 'plus', 'also', 'too', 'yet', 'still', 'though',
                'however', 'actually', 'basically', 'honestly', 'seriously',
                'right', 'okay', 'alright', 'fine', 'good', 'wait', 'look',
                'listen', 'see', 'hey', 'like', 'just', 'really', 'maybe',
                'perhaps', 'probably', 'definitely', 'certainly', 'obviously'
            ]
            
            # Define isolated pronouns that should NEVER be alone
            isolated_pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they']
            
            # Define gap limits for different types of merging
            MAX_OVERLAP_GAP = 2000      # 2 seconds max for overlap creation
            MAX_MERGE_GAP = 1500        # 1.5 seconds max for isolated pronoun merging
            MIN_OVERLAP_DURATION = 200  # Minimum 200ms overlap to be meaningful
            
            overlaps_created = 0
            pronouns_merged = 0
            
            # FIRST PASS: Fix isolated pronouns by merging them with next subtitle (with gap checking)
            logger.debug("=== FIRST PASS: Fixing isolated pronouns ===")
            for i in range(len(subs) - 1):
                current_sub = subs[i]
                next_sub = subs[i + 1]
                
                # Check if current subtitle is just an isolated pronoun
                current_text = current_sub.text.strip().lower()
                current_duration = current_sub.end.ordinal - current_sub.start.ordinal
                
                if (current_text in isolated_pronouns and 
                    current_duration < 300 and  # Very short duration
                    len(current_text.split()) == 1):  # Single word
                    
                    # Check timing gap before merging
                    current_end = current_sub.end.ordinal
                    next_start = next_sub.start.ordinal
                    gap = next_start - current_end
                    
                    logger.debug(f"Analyzing isolated pronoun '{current_sub.text}' -> '{next_sub.text}' (gap: {gap}ms)")
                    
                    if gap <= MAX_MERGE_GAP:
                        logger.info(f"Merging isolated pronoun: '{current_sub.text}' + '{next_sub.text}' (gap: {gap}ms)")
                        
                        # Merge current subtitle with next subtitle
                        merged_text = f"{current_sub.text.strip()} {next_sub.text.strip()}"
                        
                        # Use current subtitle's start time and next subtitle's end time
                        current_sub.text = merged_text
                        current_sub.end = next_sub.end
                        
                        # Mark next subtitle for deletion
                        next_sub.text = "[DELETE_ME]"
                        
                        pronouns_merged += 1
                        logger.info(f"  Merged result: '{merged_text}' ({current_sub.start} -> {current_sub.end})")
                    else:
                        logger.debug(f"  Skipping merge: gap too large ({gap}ms > {MAX_MERGE_GAP}ms)")
            
            # Remove marked subtitles
            subs = [sub for sub in subs if not sub.text.startswith("[DELETE_ME]")]
            logger.info(f"Merged {pronouns_merged} isolated pronouns")
            
            # SECOND PASS: Create overlaps for appropriate interjections (with gap checking)
            logger.debug("=== SECOND PASS: Creating overlaps for interjections ===")
            for i in range(len(subs) - 1):
                current_sub = subs[i]
                next_sub = subs[i + 1]
                
                # Skip if already marked for track assignment
                if current_sub.text.startswith('[TRACK') or next_sub.text.startswith('[TRACK'):
                    continue
                
                logger.debug(f"Analyzing subtitle pair {i+1}-{i+2}:")
                logger.debug(f"  Current: '{current_sub.text}' ({current_sub.start}-{current_sub.end})")
                logger.debug(f"  Next: '{next_sub.text}' ({next_sub.start}-{next_sub.end})")
                
                # Check timing gap first
                current_end = current_sub.end.ordinal
                next_start = next_sub.start.ordinal
                gap = next_start - current_end
                
                if gap > MAX_OVERLAP_GAP:
                    logger.debug(f"  Skipping: gap too large ({gap}ms > {MAX_OVERLAP_GAP}ms)")
                    continue
                
                # Check if current subtitle is a single short overlap candidate
                if self._is_single_overlap_candidate(current_sub, overlap_candidates):
                    logger.debug(f"  Current subtitle is a single overlap candidate")
                    
                    # Check if next subtitle is suitable for overlapping
                    if self._is_suitable_for_overlap(next_sub):
                        logger.debug(f"  Next subtitle is suitable for overlap")
                        
                        # Calculate overlap parameters
                        overlap_duration = self._calculate_single_overlap_duration_with_gap(
                            current_sub, next_sub, gap
                        )
                        
                        if overlap_duration >= MIN_OVERLAP_DURATION:
                            # Calculate new end time for current subtitle
                            next_end = next_sub.end.ordinal
                            overlap_end = next_start + overlap_duration
                            
                            # Ensure we don't extend past 70% of next subtitle's duration
                            next_duration = next_end - next_start
                            max_overlap_end = next_start + int(next_duration * 0.7)
                            overlap_end = min(overlap_end, max_overlap_end)
                            
                            # Final check that overlap is still meaningful
                            actual_overlap = overlap_end - next_start
                            if actual_overlap >= MIN_OVERLAP_DURATION:
                                logger.info(f"Creating single subtitle overlap:")
                                logger.info(f"  Subtitle {i+1}: '{current_sub.text}' -> extending to overlap")
                                logger.info(f"  Subtitle {i+2}: '{next_sub.text}' -> marking for track 2")
                                logger.info(f"  Original gap: {gap}ms -> New overlap: {actual_overlap}ms")
                                
                                # Extend current subtitle to create overlap
                                current_sub.end = pysrt.SubRipTime.from_ordinal(overlap_end)
                                
                                # Mark next subtitle for track 2 placement
                                next_sub.text = f"[TRACK2]{next_sub.text}"
                                
                                overlaps_created += 1
                            else:
                                logger.debug(f"  Final overlap too small ({actual_overlap}ms), skipping")
                        else:
                            logger.debug(f"  Calculated overlap too small ({overlap_duration}ms), skipping")
                    else:
                        logger.debug(f"  Next subtitle not suitable for overlap")
                else:
                    logger.debug(f"  Current subtitle not a single overlap candidate")
            
            # THIRD PASS: Final safety check for any remaining isolated pronouns (with strict gap limits)
            logger.debug("=== THIRD PASS: Final safety check for isolated pronouns ===")
            final_subs = []
            i = 0
            additional_merges = 0
            
            while i < len(subs):
                current_sub = subs[i]
                
                # Check if this is still an isolated pronoun with very strict criteria
                current_text = current_sub.text.strip().lower()
                current_duration = current_sub.end.ordinal - current_sub.start.ordinal
                
                if (current_text in isolated_pronouns and 
                    current_duration < 200 and  # Even stricter duration limit
                    len(current_text.split()) == 1 and
                    i < len(subs) - 1):  # Not the last subtitle
                    
                    next_sub = subs[i + 1]
                    
                    # Check timing gap with strict limit
                    current_end = current_sub.end.ordinal
                    next_start = next_sub.start.ordinal
                    gap = next_start - current_end
                    
                    # Much stricter gap limit for final safety check
                    STRICT_MERGE_GAP = 800  # Only 800ms max for final safety merging
                    
                    if gap <= STRICT_MERGE_GAP:
                        logger.warning(f"Final safety check: merging isolated pronoun '{current_sub.text}' with '{next_sub.text}' (gap: {gap}ms)")
                        
                        # Merge with next subtitle
                        merged_text = f"{current_sub.text.strip()} {next_sub.text.strip()}"
                        merged_sub = pysrt.SubRipItem(
                            index=current_sub.index,
                            start=current_sub.start,
                            end=next_sub.end,
                            text=merged_text
                        )
                        
                        final_subs.append(merged_sub)
                        additional_merges += 1
                        i += 2  # Skip the next subtitle since we merged it
                    else:
                        logger.debug(f"Final safety check: NOT merging '{current_sub.text}' - gap too large ({gap}ms > {STRICT_MERGE_GAP}ms)")
                        final_subs.append(current_sub)
                        i += 1
                else:
                    final_subs.append(current_sub)
                    i += 1
            
            # Rebuild subtitle file with proper indexing
            new_subs = pysrt.SubRipFile()
            for i, sub in enumerate(final_subs, 1):
                sub.index = i
                new_subs.append(sub)
            
            # Save the modified SRT file
            new_subs.save(output_file, encoding='utf-8')
            
            logger.info(f"Created {overlaps_created} single subtitle overlaps and fixed {pronouns_merged + additional_merges} isolated pronouns")
            logger.info(f"Gap limits used: Overlap={MAX_OVERLAP_GAP}ms, Merge={MAX_MERGE_GAP}ms, Final={800}ms")
            return True
            
        except Exception as e:
            logger.error(f"Error creating single subtitle overlaps: {str(e)}")
            logger.exception(e)
            return False

    def _calculate_single_overlap_duration_with_gap(self, current_sub, next_sub, gap):
        """
        Calculate appropriate overlap duration considering the gap between subtitles.
        
        Returns:
            int: Overlap duration in milliseconds, or 0 if no overlap should be created
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        # Get basic properties
        current_text = current_sub.text.strip()
        next_text = next_sub.text.strip()
        current_duration = current_sub.end.ordinal - current_sub.start.ordinal
        next_duration = next_sub.end.ordinal - next_sub.start.ordinal
        
        logger.debug(f"    Calculating overlap with gap: '{current_text}' + '{next_text}' (gap: {gap}ms)")
        
        # LIMITS based on gap size
        if gap > 1500:  # > 1.5 seconds
            logger.debug(f"    No overlap: gap too large for meaningful overlap ({gap}ms)")
            return 0
        elif gap > 1000:  # 1-1.5 seconds
            base_overlap = 200  # Very conservative for large gaps
            logger.debug(f"    Large gap ({gap}ms): using conservative overlap")
        elif gap > 500:   # 0.5-1 seconds
            base_overlap = 350  # Moderate overlap
            logger.debug(f"    Medium gap ({gap}ms): using moderate overlap")
        elif gap > 0:     # 0-0.5 seconds
            base_overlap = 450  # Normal overlap for small gaps
            logger.debug(f"    Small gap ({gap}ms): using normal overlap")
        else:             # Already overlapping or touching
            base_overlap = 300  # Conservative for existing overlaps
            logger.debug(f"    Already overlapping ({gap}ms): using conservative extension")
        
        # Adjust based on subtitle characteristics
        current_words = len(current_text.split())
        next_words = len(next_text.split())
        
        # Single word interjections can have slightly longer overlaps
        if current_words == 1:
            base_overlap += 50
            logger.debug(f"    +50ms for single word")
        
        # Adjust based on next subtitle length
        if next_words >= 5:
            base_overlap += 75  # Longer content can handle more overlap
            logger.debug(f"    +75ms for long next subtitle ({next_words} words)")
        elif next_words <= 2:
            base_overlap -= 50  # Short content needs less overlap
            logger.debug(f"    -50ms for short next subtitle ({next_words} words)")
        
        # Don't exceed reasonable bounds
        min_overlap = 200
        max_overlap = 600
        
        final_overlap = max(min_overlap, min(base_overlap, max_overlap))
        
        # Additional check: don't overlap more than 60% of next subtitle
        max_next_overlap = int(next_duration * 0.6)
        final_overlap = min(final_overlap, max_next_overlap)
        
        logger.debug(f"    Final overlap: {final_overlap}ms")
        return final_overlap

    def _is_single_overlap_candidate(self, subtitle, overlap_candidates):
        """
        Check if a subtitle is a candidate for single subtitle overlap.
        
        Criteria:
        1. Short text (1-3 words)
        2. Contains overlap candidate words
        3. Short duration (typically under 500ms, but can be longer for important words)
        4. Not already marked for special track placement
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        # Get basic properties
        text = subtitle.text.strip()
        words = text.split()
        duration_ms = subtitle.end.ordinal - subtitle.start.ordinal
        
        logger.debug(f"    Checking overlap candidate: '{text}' ({len(words)} words, {duration_ms}ms)")
        
        # Must be 1-3 words
        if len(words) < 1 or len(words) > 3:
            logger.debug(f"    Not candidate: {len(words)} words (need 1-3)")
            return False
        
        # Check if any word is an overlap candidate
        contains_candidate = False
        for word in words:
            clean_word = word.lower().strip('.,!?;:')
            if clean_word in overlap_candidates:
                contains_candidate = True
                logger.debug(f"    Found overlap candidate word: '{clean_word}'")
                break
        
        if not contains_candidate:
            logger.debug(f"    Not candidate: no overlap candidate words found")
            return False
        
        # Duration check - be more flexible for important words
        max_duration = 800  # Allow up to 800ms for single subtitles
        if duration_ms > max_duration:
            logger.debug(f"    Not candidate: duration {duration_ms}ms > {max_duration}ms")
            return False
        
        logger.debug(f"    IS CANDIDATE: '{text}' qualifies for single subtitle overlap")
        return True

    def _is_suitable_for_overlap(self, subtitle):
        """
        Check if a subtitle is suitable to be overlapped by a preceding single subtitle.
        
        Criteria:
        1. Reasonable length (3+ words or longer duration)
        2. Not another short interjection
        3. Not already marked for special track placement
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        # Skip if already marked
        if subtitle.text.startswith('[TRACK'):
            logger.debug(f"    Not suitable: already marked for track placement")
            return False
        
        text = subtitle.text.strip()
        words = text.split()
        duration_ms = subtitle.end.ordinal - subtitle.start.ordinal
        
        logger.debug(f"    Checking suitability: '{text}' ({len(words)} words, {duration_ms}ms)")
        
        # Should have reasonable content (3+ words or longer duration)
        if len(words) < 3 and duration_ms < 600:
            logger.debug(f"    Not suitable: too short ({len(words)} words, {duration_ms}ms)")
            return False
        
        # Should not be another short interjection
        short_interjections = ['but', 'and', 'or', 'so', 'oh', 'ah', 'yeah', 'no', 'yes', 'well']
        if len(words) <= 2:
            first_word = words[0].lower().strip('.,!?;:')
            if first_word in short_interjections:
                logger.debug(f"    Not suitable: is another short interjection '{first_word}'")
                return False
        
        logger.debug(f"    IS SUITABLE: '{text}' can be overlapped")
        return True

    def _calculate_single_overlap_duration(self, current_sub, next_sub):
        """
        Calculate appropriate overlap duration for single subtitle overlaps with reasonable limits.
        
        Returns:
            int: Overlap duration in milliseconds, or 0 if no overlap should be created
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        current_text = current_sub.text.strip()
        next_text = next_sub.text.strip()
        
        current_words = len(current_text.split())
        next_words = len(next_text.split())
        
        # Get timing information
        current_start = current_sub.start.ordinal
        current_end = current_sub.end.ordinal
        current_duration = current_end - current_start
        next_start = next_sub.start.ordinal
        next_end = next_sub.end.ordinal
        next_duration = next_end - next_start
        
        gap_duration = next_start - current_end
        
        logger.debug(f"    Calculating overlap: '{current_text}' ({current_words}w, {current_duration}ms) + '{next_text}' ({next_words}w, {next_duration}ms)")
        logger.debug(f"    Gap between subtitles: {gap_duration}ms")
        
        # LIMITS: Don't create overlaps in these cases
        
        # 1. Don't bridge gaps larger than 2 seconds (separate thoughts)
        MAX_GAP_TO_BRIDGE = 2000  # 2 seconds
        if gap_duration > MAX_GAP_TO_BRIDGE:
            logger.debug(f"    No overlap: gap too large ({gap_duration}ms > {MAX_GAP_TO_BRIDGE}ms)")
            return 0
        
        # 2. Don't extend very short subtitles too much (avoid 25x extensions)
        MAX_EXTENSION_RATIO = 4.0  # Don't extend more than 4x original duration
        max_extension_by_ratio = current_duration * MAX_EXTENSION_RATIO
        
        # 3. Absolute maximum extension (regardless of original duration)
        MAX_ABSOLUTE_EXTENSION = 1500  # 1.5 seconds max extension
        
        # 4. Don't extend into more than 60% of next subtitle's duration
        MAX_NEXT_OVERLAP_PERCENT = 0.6
        max_next_overlap = int(next_duration * MAX_NEXT_OVERLAP_PERCENT)
        
        # CALCULATION: Base overlap duration
        
        # Start with a reasonable base
        base_overlap = 400  # 400ms base
        
        # Adjust based on current subtitle characteristics
        current_word = current_text.split()[0].lower().strip('.,!?;:')
        
        # High-priority interjection words get slightly longer overlaps
        high_priority_words = ['but', 'and', 'so', 'well', 'oh', 'yeah']
        if current_word in high_priority_words:
            base_overlap += 150
            logger.debug(f"    +150ms for high-priority word '{current_word}'")
        
        # Single words can have slightly longer overlaps
        if current_words == 1:
            base_overlap += 100
            logger.debug(f"    +100ms for single word")
        
        # Adjust based on gap size
        if gap_duration <= 200:
            # Very small gap or already overlapping - minimal extension
            base_overlap = min(base_overlap, 300)
            logger.debug(f"    Limited to 300ms due to small gap ({gap_duration}ms)")
        elif gap_duration <= 500:
            # Small gap - moderate extension
            base_overlap += 50
            logger.debug(f"    +50ms for small gap ({gap_duration}ms)")
        elif gap_duration >= 1000:
            # Large gap - reduce overlap (probably separate thoughts)
            base_overlap -= 100
            logger.debug(f"    -100ms for large gap ({gap_duration}ms)")
        
        # Adjust based on next subtitle characteristics
        if next_words >= 6:
            # Longer next subtitle can handle more overlap
            base_overlap += 100
            logger.debug(f"    +100ms for long next subtitle ({next_words} words)")
        elif next_words <= 2:
            # Shorter next subtitle needs less overlap
            base_overlap -= 100
            logger.debug(f"    -100ms for short next subtitle ({next_words} words)")
        
        # APPLY LIMITS: Ensure overlap doesn't exceed any of our limits
        
        # Calculate the actual maximum overlap we can create
        max_possible_overlap = gap_duration + max_next_overlap
        
        # Apply all limits
        final_overlap = min(
            base_overlap,
            max_extension_by_ratio - current_duration,  # Don't exceed ratio limit
            MAX_ABSOLUTE_EXTENSION,                     # Don't exceed absolute limit  
            max_possible_overlap,                       # Don't exceed available space
            max_next_overlap                           # Don't overlap too much of next subtitle
        )
        
        # Ensure minimum viable overlap
        MIN_OVERLAP = 200  # Minimum 200ms to be meaningful
        if final_overlap < MIN_OVERLAP:
            logger.debug(f"    No overlap: calculated overlap too small ({final_overlap}ms < {MIN_OVERLAP}ms)")
            return 0
        
        # Log the limiting factor for debugging
        limiting_factors = []
        if final_overlap == base_overlap:
            limiting_factors.append("base calculation")
        if final_overlap == max_extension_by_ratio - current_duration:
            limiting_factors.append(f"extension ratio ({MAX_EXTENSION_RATIO}x)")
        if final_overlap == MAX_ABSOLUTE_EXTENSION:
            limiting_factors.append(f"absolute limit ({MAX_ABSOLUTE_EXTENSION}ms)")
        if final_overlap == max_possible_overlap:
            limiting_factors.append("available space")
        if final_overlap == max_next_overlap:
            limiting_factors.append(f"next subtitle overlap limit ({MAX_NEXT_OVERLAP_PERCENT*100}%)")
        
        logger.debug(f"    Final overlap: {final_overlap}ms (limited by: {', '.join(limiting_factors)})")
        logger.debug(f"    Extension: {current_duration}ms -> {current_duration + final_overlap}ms ({(final_overlap/current_duration)*100:.1f}% increase)")
        
        return final_overlap

    def assign_tracks_to_subtitles(self, subs):
        """
        Assign subtitles to tracks ensuring overlaps only happen in pairs, never allowing
        three subtitles to display simultaneously. Enhanced to handle both [TRACK2] and [TRACK3] markers.
        """
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Assigning {len(subs)} subtitles to tracks with improved overlap control")
        
        # Check if we have any special track markers
        has_track2 = any(text.startswith('[TRACK2]') for _, _, text in subs)
        has_track3 = any(text.startswith('[TRACK3]') for _, _, text in subs)
        
        if has_track2 or has_track3:
            # Separate regular subtitles from specially marked ones
            regular_subs = []
            track2_subs = []
            track3_subs = []
            original_indices = []
            track2_indices = []
            track3_indices = []
            
            for i, (start, duration, text) in enumerate(subs):
                if text.startswith('[TRACK2]'):
                    # Remove marker and store for track 2
                    clean_text = text[8:]  # Remove '[TRACK2]' prefix
                    track2_subs.append((start, duration, clean_text))
                    track2_indices.append(i)
                elif text.startswith('[TRACK3]'):
                    # Remove marker and store for track 3
                    clean_text = text[8:]  # Remove '[TRACK3]' prefix
                    track3_subs.append((start, duration, clean_text))
                    track3_indices.append(i)
                else:
                    regular_subs.append((start, duration, text))
                    original_indices.append(i)
            
            # Apply original track assignment logic to regular subtitles ONLY
            track_assignments, assigned_tracks = self._assign_tracks_original_logic(regular_subs)
            
            # Create final track assignments including special tracks
            final_assigned_tracks = [None] * len(subs)
            
            # Fill in regular subtitle track assignments
            for i, track in enumerate(assigned_tracks):
                original_idx = original_indices[i]
                final_assigned_tracks[original_idx] = track
            
            # Assign track 2 to marked subtitles
            for i, original_idx in enumerate(track2_indices):
                final_assigned_tracks[original_idx] = 2
                logger.info(f"Assigned [TRACK2] marked subtitle {original_idx + 1} ('{track2_subs[i][2]}') to track 2")
            
            # Assign track 3 to marked subtitles
            for i, original_idx in enumerate(track3_indices):
                final_assigned_tracks[original_idx] = 3
                logger.info(f"Assigned [TRACK3] marked subtitle {original_idx + 1} ('{track3_subs[i][2]}') to track 3")
            
            # Count final assignments
            track1_count = final_assigned_tracks.count(1)
            track2_count = final_assigned_tracks.count(2) 
            track3_count = final_assigned_tracks.count(3)
            
            logger.info(f"Final track assignment: Track 1: {track1_count}, Track 2: {track2_count}, Track 3: {track3_count}")
            
            return track_assignments, final_assigned_tracks
        
        else:
            # No special track markers, use original logic
            return self._assign_tracks_original_logic(subs)

    def _smart_split_text_for_gaming(self, text, max_chars_per_line=25):
        """
        Split text specifically optimized for gaming videos where text needs to be large and readable.
        This is more aggressive than the original method but maintains logical flow.
        """
        logger = logging.getLogger('SubtitleGenerator')
        words = text.split()
        
        # For gaming videos, we want shorter segments (3-5 words max per line)
        if len(words) <= 4:
            logger.debug(f"Text short enough for gaming: {len(words)} words")
            return [text]
        
        logger.debug(f"Gaming split: '{text}' ({len(words)} words, {len(text)} chars)")
        
        # Define logical break points specifically for gaming (order of preference)
        gaming_break_patterns = [
            # 1. CONJUNCTIONS - Always good break points for gaming
            {'words': ['but', 'and', 'or', 'so', 'yet'], 'priority': 10, 'break_before': True},
            
            # 2. SEQUENCE WORDS - Natural story progression breaks
            {'words': ['then', 'now', 'next', 'after', 'before', 'finally'], 'priority': 9, 'break_before': True},
            
            # 3. TRANSITION WORDS - Good for dramatic effect
            {'words': ['however', 'meanwhile', 'suddenly', 'actually', 'basically'], 'priority': 8, 'break_before': True},
            
            # 4. PRONOUNS - Start of new action/subject
            {'words': ['i', 'you', 'he', 'she', 'we', 'they', 'it'], 'priority': 7, 'break_before': True},
            
            # 5. QUESTION WORDS - Natural break points
            {'words': ['what', 'where', 'when', 'why', 'how', 'who'], 'priority': 6, 'break_before': True},
            
            # 6. ACTION WORDS - Verbs that indicate new actions
            {'words': ['go', 'come', 'get', 'take', 'make', 'do', 'see', 'look', 'find'], 'priority': 5, 'break_before': True},
            
            # 7. PREPOSITIONS - Can break longer phrases
            {'words': ['in', 'on', 'at', 'to', 'from', 'with', 'by', 'for'], 'priority': 4, 'break_before': True},
            
            # 8. ARTICLES - Last resort but still logical
            {'words': ['the', 'a', 'an'], 'priority': 3, 'break_before': True}
        ]
        
        # Find all potential break points
        break_candidates = []
        
        for i, word in enumerate(words):
            if i == 0 or i >= len(words) - 1:  # Don't break at very start or end
                continue
                
            word_lower = word.lower().strip('.,!?;:')
            
            # Check against all patterns
            for pattern in gaming_break_patterns:
                if word_lower in pattern['words']:
                    # Calculate how balanced this split would be
                    left_words = i if pattern['break_before'] else i + 1
                    right_words = len(words) - left_words
                    
                    # Prefer more balanced splits but allow 2-6 words per segment
                    if 2 <= left_words <= 6 and 2 <= right_words <= 6:
                        balance_score = 10 - abs(left_words - right_words)  # Higher score for better balance
                        total_score = pattern['priority'] + balance_score
                        
                        break_candidates.append({
                            'position': left_words,
                            'word': word,
                            'pattern_type': pattern['words'][0] if len(pattern['words']) == 1 else 'multi',
                            'score': total_score,
                            'left_words': left_words,
                            'right_words': right_words
                        })
                        
                        logger.debug(f"Break candidate at word {left_words}: '{word}' "
                                f"(type: {pattern['words'][0] if len(pattern['words']) == 1 else 'multi'}, "
                                f"left: {left_words}, right: {right_words}, score: {total_score})")
        
        # Choose the best break point
        if break_candidates:
            # Sort by score (highest first)
            break_candidates.sort(key=lambda x: x['score'], reverse=True)
            best_break = break_candidates[0]
            
            logger.info(f"Gaming split using '{best_break['word']}' at position {best_break['position']} "
                    f"(score: {best_break['score']}, type: {best_break['pattern_type']})")
            
            # Split at the best position
            left_segment = ' '.join(words[:best_break['position']]).strip()
            right_segment = ' '.join(words[best_break['position']:]).strip()
            
            logger.info(f"Gaming split result:")
            logger.info(f"  Left: '{left_segment}' ({len(left_segment.split())} words)")
            logger.info(f"  Right: '{right_segment}' ({len(right_segment.split())} words)")
            
            # Recursively split if segments are still too long
            result = []
            
            # Split left segment if needed
            if len(left_segment.split()) > 5:
                result.extend(self._smart_split_text_for_gaming(left_segment, max_chars_per_line))
            else:
                result.append(left_segment)
            
            # Split right segment if needed
            if len(right_segment.split()) > 5:
                result.extend(self._smart_split_text_for_gaming(right_segment, max_chars_per_line))
            else:
                result.append(right_segment)
            
            return result
        
        # If no logical breaks found, force split at reasonable points
        logger.debug("No logical breaks found, using forced gaming split")
        return self._force_gaming_split(words)

    def _force_gaming_split(self, words):
        """
        Force split for gaming when no logical breaks are found.
        Aims for 3-4 words per segment maximum.
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        total_words = len(words)
        logger.debug(f"Force gaming split: {total_words} words")
        
        # For gaming, prefer 3-4 words per segment
        target_segment_size = 4
        
        if total_words <= target_segment_size:
            return [' '.join(words)]
        
        # Calculate number of segments needed
        num_segments = (total_words + target_segment_size - 1) // target_segment_size
        words_per_segment = total_words // num_segments
        
        # Ensure minimum 2 words per segment
        if words_per_segment < 2:
            words_per_segment = 2
            num_segments = (total_words + 1) // 2
        
        logger.debug(f"Creating {num_segments} segments with ~{words_per_segment} words each")
        
        segments = []
        start_idx = 0
        
        for i in range(num_segments):
            if i == num_segments - 1:  # Last segment gets remaining words
                end_idx = total_words
            else:
                end_idx = start_idx + words_per_segment
                
                # Try to avoid splitting at bad points
                if end_idx < total_words - 1:
                    current_word = words[end_idx].lower().strip('.,!?;:')
                    
                    # Don't split right before these words
                    avoid_splitting_before = ['the', 'a', 'an', 'of', 'in', 'on', 'at']
                    if current_word in avoid_splitting_before and end_idx > start_idx + 1:
                        end_idx -= 1  # Move split point back one word
                    
                    # Don't split right after these words
                    prev_word = words[end_idx - 1].lower().strip('.,!?;:')
                    avoid_splitting_after = ['the', 'a', 'an', 'very', 'really', 'so', 'too']
                    if prev_word in avoid_splitting_after and end_idx < total_words - 1:
                        end_idx += 1  # Move split point forward one word
            
            segment = ' '.join(words[start_idx:end_idx])
            segments.append(segment)
            logger.debug(f"Segment {i+1}: '{segment}' ({len(segment.split())} words)")
            
            start_idx = end_idx
        
        logger.info(f"Force gaming split created {len(segments)} segments")
        return segments

    def split_long_track_marked_subtitles(self, input_file, output_file):
        """
        Gaming-optimized version: Split long subtitles while preserving track markers.
        Much more aggressive splitting suitable for gaming videos.
        """
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Gaming split: Splitting long track-marked subtitles from {input_file} to {output_file}")
        
        try:
            import pysrt
            
            # Load SRT file
            subs = pysrt.open(input_file, encoding='utf-8')
            logger.debug(f"Loaded {len(subs)} subtitles for gaming track-marked splitting")
            
            new_subs = pysrt.SubRipFile()
            counter = 1
            
            for sub_index, sub in enumerate(subs):
                logger.debug(f"Processing subtitle {sub_index+1}/{len(subs)}: {sub.text}")
                
                # Skip empty subtitles
                if not sub.text.strip():
                    continue
                
                # Check if this subtitle has a track marker
                track_marker = None
                clean_text = sub.text
                
                if sub.text.startswith('[TRACK2]'):
                    track_marker = '[TRACK2]'
                    clean_text = sub.text[8:]  # Remove '[TRACK2]' prefix
                elif sub.text.startswith('[TRACK3]'):
                    track_marker = '[TRACK3]'
                    clean_text = sub.text[8:]  # Remove '[TRACK3]' prefix
                
                logger.debug(f"Track marker: {track_marker}, Clean text: '{clean_text}'")
                
                # GAMING THRESHOLDS: More aggressive for gaming videos
                words = clean_text.split()
                char_count = len(clean_text)
                
                # Split if more than 5 words OR more than 30 characters (gaming-optimized)
                needs_splitting = len(words) > 5 or char_count > 30
                
                if not needs_splitting or len(words) <= 2:
                    # No splitting needed or too short to split safely
                    new_subs.append(pysrt.SubRipItem(
                        index=counter,
                        start=sub.start,
                        end=sub.end,
                        text=sub.text.strip()  # Keep original text with marker
                    ))
                    counter += 1
                    continue
                
                logger.info(f"Gaming split needed: '{clean_text}' ({len(words)} words, {char_count} chars)")
                
                # Use gaming-optimized splitting
                segments = self._smart_split_text_for_gaming(clean_text, max_chars_per_line=25)
                logger.debug(f"Gaming split into {len(segments)} segments: {segments}")
                
                if len(segments) <= 1:
                    # Splitting failed, keep as is
                    new_subs.append(pysrt.SubRipItem(
                        index=counter,
                        start=sub.start,
                        end=sub.end,
                        text=sub.text.strip()
                    ))
                    counter += 1
                    continue
                
                # Calculate timing for each segment
                start_ms = sub.start.ordinal
                total_duration = sub.end.ordinal - sub.start.ordinal
                
                # For gaming, give each segment equal time (simpler and more predictable)
                segment_duration = total_duration // len(segments)
                current_start = start_ms
                
                for i, segment in enumerate(segments):
                    if i == len(segments) - 1:
                        # Last segment gets remaining time
                        segment_end = sub.end.ordinal
                    else:
                        segment_end = current_start + segment_duration
                    
                    # Restore the track marker if it existed
                    final_text = f"{track_marker}{segment}" if track_marker else segment
                    
                    logger.debug(f"Gaming segment {i+1}/{len(segments)}: '{final_text}' ({current_start}-{segment_end})")
                    
                    new_subs.append(pysrt.SubRipItem(
                        index=counter,
                        start=pysrt.SubRipTime.from_ordinal(current_start),
                        end=pysrt.SubRipTime.from_ordinal(segment_end),
                        text=final_text
                    ))
                    counter += 1
                    current_start = segment_end
            
            # Save the new subtitles
            new_subs.save(output_file, encoding='utf-8')
            logger.info(f"Gaming split: Created {len(new_subs)} total subtitles optimized for gaming videos")
            return True
            
        except Exception as e:
            logger.error(f"Error in gaming split: {str(e)}")
            logger.exception(e)
            return False

    def _smart_split_text(self, text):
        """
        Intelligently split text respecting natural language boundaries.
        MUCH better than the old mechanical word-counting approach.
        """
        logger = logging.getLogger('SubtitleGenerator')
        words = text.split()
        
        # Don't split short texts
        if len(words) <= 6:
            logger.debug(f"Text too short to split: {len(words)} words")
            return [text]
        
        logger.debug(f"Smart splitting: '{text}' ({len(words)} words)")
        
        # Define natural break points (in order of preference)
        break_patterns = [
            # 0. HIGHEST PRIORITY: Smart punctuation splits that prevent isolation
            # This finds punctuation followed by 3+ words (prevents isolation)
            {'pattern': r'([.!?;:])\s+(?=\w+(?:\s+\w+){2,})', 'priority': 15, 'type': 'smart_punctuation'},
            
            # 1. After complete clauses ending with punctuation
            {'pattern': r'[.!?]\s+', 'priority': 10, 'type': 'punctuation'},
            
            # 2. Before coordinating conjunctions (but, and, or, so)
            {'pattern': r'\s+(?=\b(?:but|and|or|so|yet)\b)', 'priority': 8, 'type': 'conjunction'},
            
            # 3. After subordinating conjunctions + clause
            {'pattern': r'(?<=\b(?:because|since|although|though|while|when|if|unless)\b[^.!?]{8,20})\s+', 'priority': 7, 'type': 'subordinate'},
            
            # 4. Before infinitive phrases (to + verb)
            {'pattern': r'\s+(?=\bto\s+\w+)', 'priority': 6, 'type': 'infinitive'},
            
            # 5. After prepositional phrases
            {'pattern': r'(?<=\b(?:in|on|at|by|for|with|from|about|through|during)\b\s+\w+(?:\s+\w+)?)\s+', 'priority': 5, 'type': 'preposition'},
            
            # 6. Before question words (what, where, how, why, when)
            {'pattern': r'\s+(?=\b(?:what|where|how|why|when|who)\b)', 'priority': 4, 'type': 'question'},
        ]
        
        # Find all potential break points
        break_points = []
        for pattern_info in break_patterns:
            import re
            for match in re.finditer(pattern_info['pattern'], text, re.IGNORECASE):
                break_pos = match.start() if 'before' in str(pattern_info['pattern']) else match.end()
                
                # Convert position to word index
                word_index = len(text[:break_pos].split())
                
                # Only consider breaks that create reasonable segments
                if 2 <= word_index <= len(words) - 2:  # Don't break too early or too late
                    break_points.append({
                        'word_index': word_index,
                        'priority': pattern_info['priority'],
                        'type': pattern_info['type'],
                        'position': break_pos
                    })
                    logger.debug(f"Found {pattern_info['type']} break at word {word_index}: '{text[max(0,break_pos-10):break_pos+10]}'")
        
        # If no natural breaks found, use smart fallback
        if not break_points:
            logger.debug("No natural breaks found, using smart fallback")
            return self._fallback_split(words)
        
        # Choose the best break point
        # Prefer higher priority breaks that create balanced segments
        best_break = None
        best_score = -1
        
        for bp in break_points:
            # Calculate balance score (prefer splits that create roughly equal segments)
            left_words = bp['word_index']
            right_words = len(words) - bp['word_index']
            balance_penalty = abs(left_words - right_words) * 0.5
            
            # Final score = priority - balance_penalty
            score = bp['priority'] - balance_penalty
            
            logger.debug(f"Break at word {bp['word_index']} ({bp['type']}): "
                        f"left={left_words}, right={right_words}, "
                        f"priority={bp['priority']}, balance_penalty={balance_penalty:.1f}, "
                        f"final_score={score:.1f}")
            
            if score > best_score:
                best_score = score
                best_break = bp
        
        if best_break:
            # Split at the best break point
            left_segment = ' '.join(words[:best_break['word_index']]).strip()
            right_segment = ' '.join(words[best_break['word_index']:]).strip()
            
            logger.info(f"Smart split using {best_break['type']} break (score: {best_score:.1f}):")
            logger.info(f"  Left: '{left_segment}' ({len(left_segment.split())} words)")
            logger.info(f"  Right: '{right_segment}' ({len(right_segment.split())} words)")
            
            # Recursively split if segments are still too long
            result = []
            if len(left_segment.split()) > 6:
                result.extend(self._smart_split_text(left_segment))
            else:
                result.append(left_segment)
                
            if len(right_segment.split()) > 6:
                result.extend(self._smart_split_text(right_segment))
            else:
                result.append(right_segment)
                
            return result
        
        # Fallback if no good breaks
        logger.debug("No good natural breaks found, using fallback")
        return self._fallback_split(words)

    def _fallback_split(self, words):
        """
        Fallback splitting method when no natural breaks are found.
        Still tries to avoid isolated words.
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        total_words = len(words)
        
        # For medium lengths, split roughly in half but avoid isolation
        if total_words <= 10:
            # Split closer to 60/40 to avoid tiny segments
            split_point = max(3, total_words * 3 // 5)  # Roughly 60% for first segment
        else:
            # For very long texts, split into roughly equal chunks of 5-6 words each
            split_point = total_words // 2
        
        # Ensure minimum segment sizes
        split_point = max(3, min(split_point, total_words - 3))
        
        left_segment = ' '.join(words[:split_point])
        right_segment = ' '.join(words[split_point:])
        
        logger.info(f"Fallback split at word {split_point}:")
        logger.info(f"  Left: '{left_segment}' ({len(words[:split_point])} words)")
        logger.info(f"  Right: '{right_segment}' ({len(words[split_point:])} words)")
        
        return [left_segment, right_segment]

    def _split_text_into_segments(self, text, max_words=4):
        """
        Split text into segments with a maximum number of words per segment.
        Tries to split at natural boundaries.
        """
        words = text.split()
        
        if len(words) <= max_words:
            return [text]
        
        segments = []
        current_segment = []
        
        for word in words:
            current_segment.append(word)
            
            # Check if we should split here
            if len(current_segment) >= max_words:
                # Try to find a good split point
                segment_text = ' '.join(current_segment)
                
                # Look for natural split points in the last few words
                if len(current_segment) >= 3:
                    # Check if we can split before the last word or two
                    for split_point in range(len(current_segment) - 1, max(0, len(current_segment) - 3), -1):
                        if split_point >= 2:  # Ensure minimum segment size
                            segment_text = ' '.join(current_segment[:split_point])
                            segments.append(segment_text)
                            current_segment = current_segment[split_point:]
                            break
                    else:
                        # No good split point, use the whole segment
                        segments.append(segment_text)
                        current_segment = []
                else:
                    segments.append(segment_text)
                    current_segment = []
        
        # Add any remaining words
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments

    def add_subtitles_for_file(self, processed_srt_path, frame_rate, timeline, project, target_track_start, template):
        """Add subtitles for a specific file to specific tracks using the specified template - FIXED to clean track markers"""
        logger.info(f"Adding subtitles from {os.path.basename(processed_srt_path)} starting at track {target_track_start}")
        template_name = template.GetClipProperty()['Clip Name']
        logger.info(f"Using template: {template_name}")
        
        try:
            # Get subtitle delay from UI
            try:
                subtitle_delay_seconds = float(self.delay_var.get())
                logger.info(f"Using subtitle delay: {subtitle_delay_seconds} seconds")
            except ValueError:
                logger.warning("Invalid delay value, using default 0.2 seconds")
                subtitle_delay_seconds = 0.2
            
            # Convert delay to frames
            delay_frames = int(subtitle_delay_seconds * frame_rate)
            logger.info(f"Subtitle delay: {subtitle_delay_seconds}s = {delay_frames} frames")
            
            media_pool = project.GetMediaPool()
            
            # Parse SRT file
            subs = []
            try:
                logger.debug("Opening SRT file for parsing")
                with open(processed_srt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                logger.debug(f"Read {len(lines)} lines from SRT file")
                
                i = 0
                while i < len(lines):
                    try:
                        # Skip empty lines
                        while i < len(lines) and not lines[i].strip():
                            i += 1
                        
                        if i >= len(lines):
                            break
                            
                        # Find index line (a number)
                        if not lines[i].strip().isdigit():
                            logger.debug(f"Line {i+1} is not a subtitle index: {lines[i].strip()}")
                            i += 1
                            continue
                        
                        subtitle_index = lines[i].strip()
                        
                        # Get timestamp line
                        i += 1
                        if i >= len(lines):
                            logger.warning(f"Unexpected end of file after subtitle index {subtitle_index}")
                            break
                            
                        timestamp_line = lines[i].strip()
                        if " --> " not in timestamp_line:
                            logger.warning(f"Invalid timestamp format for subtitle {subtitle_index}: {timestamp_line}")
                            i += 1
                            continue
                        
                        start_time, end_time = timestamp_line.split(" --> ")
                        
                        # Get text (could be multiple lines)
                        i += 1
                        text_lines = []
                        while i < len(lines) and lines[i].strip():
                            text_lines.append(lines[i].strip())
                            i += 1
                        
                        text = " ".join(text_lines)
                        
                        # Convert to frames
                        start_frames = self.time_to_frames(start_time, frame_rate)
                        end_frames = self.time_to_frames(end_time, frame_rate)
                        
                        # Apply delay to both start and end times
                        start_frames_delayed = start_frames + delay_frames
                        end_frames_delayed = end_frames + delay_frames
                        
                        # Calculate position and duration
                        timeline_pos = timeline.GetStartFrame() + start_frames_delayed
                        duration = end_frames_delayed - start_frames_delayed
                        
                        # Ensure minimum duration
                        if duration < 1:
                            duration = 1
                            
                        subs.append((timeline_pos, duration, text))
                    except Exception as e:
                        logger.error(f"Error processing subtitle at line {i}: {str(e)}")
                        i += 1
                        
                if not subs:
                    logger.error("No valid subtitles found in SRT file")
                    return False
                
                logger.info(f"Successfully parsed {len(subs)} subtitles with {subtitle_delay_seconds}s delay")
                    
            except Exception as e:
                logger.error(f"Error parsing SRT file: {str(e)}", exc_info=True)
                return False
            
            # Assign subtitles to tracks (now with track 3 support)
            logger.debug("Assigning subtitles to tracks")
            track_assignments, assigned_tracks = self.assign_tracks_to_subtitles(subs)
            
            # Get the actual number of tracks used
            unique_tracks = set(assigned_tracks)
            logger.info(f"Assigned subtitles to {len(unique_tracks)} tracks: {sorted(unique_tracks)}")
            
            # Prepare timeline tracks - ensure we have enough tracks for this file
            current_track_count = timeline.GetTrackCount("video")
            logger.debug(f"Current video track count: {current_track_count}")
            
            # Calculate how many tracks we need for this file (including potential track 3)
            max_track_needed = target_track_start + max(assigned_tracks) - 1
            tracks_to_add = max_track_needed - current_track_count
            
            if tracks_to_add > 0:
                logger.debug(f"Adding {tracks_to_add} tracks for this file (including potential track 3)")
                for _ in range(tracks_to_add):
                    timeline.AddTrack("video")
            
            # Adjust track assignments to actual timeline tracks
            final_assigned_tracks = []
            for track in assigned_tracks:
                final_track = target_track_start + track - 1
                final_assigned_tracks.append(final_track)
            
            logger.info(f"File will use tracks {min(final_assigned_tracks)} to {max(final_assigned_tracks)}")
            
            # Add subtitle clips to timeline using the specified template
            subtitle_clips = []
            for i, (timeline_pos, duration, text) in enumerate(subs):
                track_num = final_assigned_tracks[i]
                
                newClip = {
                    "mediaPoolItem": template,  # Use the specified template for this file
                    "startFrame": 0,
                    "endFrame": duration,
                    "trackIndex": track_num,
                    "recordFrame": timeline_pos
                }
                subtitle_clips.append(newClip)
            
            logger.info(f"Attempting to add {len(subtitle_clips)} clips to timeline using template '{template_name}'")
            success = media_pool.AppendToTimeline(subtitle_clips)
            
            if not success:
                logger.error("Failed to add clips to timeline")
                return False
            
            logger.info("Successfully added clips to timeline")

            # Organize subtitles by track for text updating
            subs_by_track = {}
            for idx, (timeline_pos, duration, text) in enumerate(subs):
                track_num = final_assigned_tracks[idx]
                if track_num not in subs_by_track:
                    subs_by_track[track_num] = []
                subs_by_track[track_num].append(text)

            # Initialize tracking for text similarity and size variation
            prev_texts_by_track = {}
            current_pattern_index_by_track = {}
            prev_sizes_by_track = {}  # Track previous sizes for track 3 inheritance
            size_pattern = [1.4, 0.7]  # Alternating size multipliers
            
            logger.info("Updating subtitle text...")
            for track_num, texts in sorted(subs_by_track.items()):
                try:
                    logger.debug(f"Processing track {track_num} with {len(texts)} subtitles")
                    
                    # Initialize for this track if not already done
                    if track_num not in prev_texts_by_track:
                        prev_texts_by_track[track_num] = []
                    if track_num not in current_pattern_index_by_track:
                        current_pattern_index_by_track[track_num] = 0
                    if track_num not in prev_sizes_by_track:
                        prev_sizes_by_track[track_num] = []
                        
                    sub_list = timeline.GetItemListInTrack('video', track_num)
                    if not sub_list:
                        logger.warning(f"No items found in track {track_num}")
                        continue
                        
                    logger.debug(f"Found {len(sub_list)} items in track {track_num}")
                    sub_list.sort(key=lambda clip: clip.GetStart())
                    
                    for i, clip in enumerate(sub_list):
                        if i < len(texts):
                            logger.debug(f"Processing clip {i+1}/{len(sub_list)} in track {track_num}")
                            clip.SetClipColor('Orange')
                            text = texts[i]
                            
                            # FIXED: Clean ALL track markers before displaying
                            display_text = text
                            is_track2_text = False
                            is_track3_text = False
                            
                            if text.startswith('[TRACK2]'):
                                display_text = text[8:]  # Remove '[TRACK2]' prefix
                                is_track2_text = True
                                logger.debug(f"Cleaned [TRACK2] marker: '{text}' -> '{display_text}'")
                            elif text.startswith('[TRACK3]'):
                                display_text = text[8:]  # Remove '[TRACK3]' prefix  
                                is_track3_text = True
                                logger.debug(f"Cleaned [TRACK3] marker: '{text}' -> '{display_text}'")
                            
                            # Check for similarity with previous subtitles
                            is_similar = False
                            for j, prev_text in enumerate(prev_texts_by_track[track_num][-3:]):
                                similarity = self.calculate_text_similarity(display_text, prev_text)
                                if similarity >= 0.7:
                                    is_similar = True
                                    break
                            
                            # Add this text to previous texts for future comparisons
                            prev_texts_by_track[track_num].append(display_text)
                            
                            # Format text
                            max_length = 18
                            max_size = 0.12
                            words = display_text.split()
                            current_line = ""
                            lines_formatted = []
                            
                            for word in words:
                                if len(current_line) + len(word) + 1 <= max_length:
                                    current_line += word + " "
                                else:
                                    lines_formatted.append(current_line.strip())
                                    current_line = word + " "
                            if current_line:
                                lines_formatted.append(current_line.strip())
                            
                            # Calculate text size
                            char_count = len(display_text.replace(" ", ""))
                            starting_size = 0.08
                            size_increase = max(0, 6 - char_count) * 0.1
                            new_size = min(starting_size + size_increase, max_size)
                            
                            # Apply size variation for similar subtitles
                            if is_similar:
                                current_idx = current_pattern_index_by_track[track_num]
                                size_multiplier = size_pattern[current_idx]
                                new_size = new_size * size_multiplier
                                current_pattern_index_by_track[track_num] = (current_idx + 1) % len(size_pattern)
                            else:
                                current_pattern_index_by_track[track_num] = 0
                            
                            # Ensure size stays within bounds
                            new_size = max(min(new_size, 0.18), 0.05)
                            
                            # Store size for potential track 3 inheritance
                            prev_sizes_by_track[track_num].append(new_size)
                            
                            # Special handling for track 3 size inheritance
                            final_size = new_size
                            if is_track3_text:
                                # Try to inherit size from the immediately previous subtitle (track 2)
                                target_track = track_num - 1  # Previous track (should be track 2)
                                
                                if target_track in prev_sizes_by_track and prev_sizes_by_track[target_track]:
                                    # Get the most recent size from the previous track
                                    inherited_size = prev_sizes_by_track[target_track][-1]
                                    # Use the exact same size as the previous segment for consistency
                                    final_size = inherited_size
                                    logger.info(f"Track 3 inheriting exact size {inherited_size} from track {target_track}")
                                else:
                                    # Fallback: use the calculated size but ensure it's reasonable
                                    final_size = max(0.08, min(new_size, 0.12))
                                    logger.info(f"Track 3 using fallback size {final_size} (calculated: {new_size})")
                            
                            # Update text in Fusion composition
                            comp = clip.GetFusionCompByIndex(1)
                            if comp is not None:
                                tools = comp.GetToolList()
                                
                                for tool_id, tool in tools.items():
                                    tool_name = tool.GetAttrs()['TOOLS_Name']
                                    
                                    if tool_name == 'Template':
                                        comp.SetActiveTool(tool)
                                        
                                        # Enhanced positioning logic for 3 tracks
                                        # Track positioning relative to this file's start track
                                        track_relative_to_start = track_num - target_track_start
                                        
                                        if track_relative_to_start == 0:  # Track 1 (relative)
                                            # Lower position - no newlines
                                            logger.debug(f"Track {track_num}: Lower position (track 1 relative)")
                                            tool.SetInput('StyledText', display_text)
                                        elif track_relative_to_start == 1:  # Track 2 (relative)
                                            # Upper position - with newlines
                                            logger.debug(f"Track {track_num}: Upper position (track 2 relative)")
                                            tool.SetInput('StyledText', "\n\n" + display_text)
                                        elif track_relative_to_start == 2:  # Track 3 (relative)
                                            # Right-aligned position using spaces
                                            logger.debug(f"Track {track_num}: Right-aligned position (track 3 relative)")
                                            
                                            # Calculate spacing for right alignment
                                            base_spacing = 15  # Base spacing
                                            
                                            # Try to get previous segment's text length for better alignment
                                            prev_text_length = 0
                                            if i > 0 and track_num - 1 in subs_by_track:
                                                # Look for previous track's current subtitle
                                                prev_track_texts = subs_by_track[track_num - 1]
                                                if i < len(prev_track_texts):
                                                    prev_text_length = len(prev_track_texts[i])
                                            
                                            # Adjust spacing based on previous text length and current text length
                                            length_adjustment = max(0, prev_text_length - len(display_text))
                                            total_spacing = base_spacing + (length_adjustment // 2)
                                            
                                            # Ensure reasonable spacing bounds
                                            total_spacing = max(5, min(total_spacing, 30))
                                            
                                            right_aligned_text = " " * total_spacing + display_text
                                            
                                            logger.debug(f"Track 3 text: '{display_text}' -> spacing: {total_spacing} chars")
                                            tool.SetInput('StyledText', right_aligned_text)
                                        else:
                                            # For any additional tracks, alternate between upper and lower
                                            is_upper_position = track_relative_to_start % 2 != 0
                                            if is_upper_position:
                                                logger.debug(f"Track {track_num}: Upper position (track {track_relative_to_start + 1} relative)")
                                                tool.SetInput('StyledText', "\n\n" + display_text)
                                            else:
                                                logger.debug(f"Track {track_num}: Lower position (track {track_relative_to_start + 1} relative)")
                                                tool.SetInput('StyledText', display_text)
                                            
                                        tool.SetInput('Size', final_size)
                                        logger.debug(f"Text and size set successfully")
                            
                            clip.SetClipColor('Teal')
                            logger.debug(f"Clip {i+1} updated successfully")
                except Exception as e:
                    logger.error(f"Error updating subtitles in track {track_num}: {str(e)}", exc_info=True)
            
            logger.info(f"Successfully processed file: {os.path.basename(processed_srt_path)} with template '{template_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error in add_subtitles_for_file: {str(e)}", exc_info=True)
            return False

    def preprocess_srt(self):
        """Apply selected preprocessing to the SRT file with enhanced contraction protection and single subtitle overlap creation"""
        logger = logging.getLogger('SubtitleGenerator')
        logger.info("Starting smart SRT preprocessing with contraction protection, single subtitle overlap enhancement and track-marked splitting")
        
        if not self.srt_file_path:
            logger.error("No SRT file selected")
            return False
            
        try:
            self.status_label.config(text="Preprocessing SRT file...")
            self.root.update()  # Update UI
            
            base_file = os.path.splitext(self.srt_file_path)[0]
            temp_file = f"{base_file}_temp.srt"
            logger.debug(f"Base file path: {base_file}")
            
            # Make a copy of the original file
            import shutil
            logger.debug(f"Creating temporary copy: {temp_file}")
            shutil.copy2(self.srt_file_path, temp_file)
            
            # Apply preprocessing in improved logical order:
            
            # 0. Smart comma-to-period replacement (BEFORE everything else)
            if hasattr(self, 'smart_comma_mode_var') and self.smart_comma_mode_var.get():
                logger.info("Using smart comma-to-period replacement")
                output_file = f"{base_file}_comma_replaced.srt"
                if self.replace_commas_with_periods(temp_file, output_file):
                    temp_file = output_file
                else:
                    logger.warning("Smart comma replacement failed, falling back to original method")
                    output_file = f"{base_file}_first_commas_replaced.srt"
                    self.replace_first_commas_with_periods(temp_file, output_file)
                    temp_file = output_file
            else:
                logger.info("Using original first comma replacement")
                output_file = f"{base_file}_first_commas_replaced.srt"
                self.replace_first_commas_with_periods(temp_file, output_file)
                temp_file = output_file
            
            # 0.5. Add punctuation after interjections (NEW FEATURE)
            if hasattr(self, 'interjection_punctuation_var') and self.interjection_punctuation_var.get():
                logger.info("Adding punctuation after interjections")
                output_file = f"{base_file}_interjection_punctuation.srt"
                self.add_punctuation_after_interjections(temp_file, output_file)
                temp_file = output_file
            else:
                logger.debug("Skipping interjection punctuation (not selected)")
            
            # 1. First add commas before interjections
            if self.add_commas_var.get():
                logger.info("Adding commas before interjections")
                output_file = f"{base_file}_added_commas_before.srt"
                self.add_commas_before_interjections(temp_file, output_file)
                temp_file = output_file
            else:
                logger.debug("Skipping adding commas before interjections (not selected)")
            
            # 2. Add commas after starting interjections
            if self.add_commas_after_var.get():
                logger.info("Adding commas after starting interjections")
                output_file = f"{base_file}_added_commas_after.srt"
                self.add_commas_after_starting_interjections(temp_file, output_file)
                temp_file = output_file
            else:
                logger.debug("Skipping adding commas after starting interjections (not selected)")
            
            # 3. Use linguistic segmentation (spaCy) - NOW WITH CONTRACTION PROTECTION
            if self.linguistic_segment_var.get():
                logger.info("Using linguistic segmentation with spaCy and contraction protection")
                
                # Initialize the subtitle segmenter if not already done
                if not hasattr(self, 'subtitle_segmenter') or self.subtitle_segmenter is None:
                    logger.debug("Initializing new SubtitleSegmenter instance with contraction protection")
                    self.subtitle_segmenter = SubtitleSegmenter()
                
                # Get target word length from UI
                try:
                    min_words = int(self.min_words_var.get())
                    max_words = int(self.max_words_var.get())
                except (ValueError, AttributeError):
                    logger.warning("Invalid word length values, using defaults")
                    min_words, max_words = 4, 6
                
                target_length = (min_words, max_words)
                logger.debug(f"Using target word length: {target_length}")
                
                # Process with linguistic segmentation (now includes contraction protection)
                output_file = f"{base_file}_linguistic_segmented.srt"
                success = self.subtitle_segmenter.process_srt_file(temp_file, output_file, target_length)
                
                if success:
                    logger.info(f"Successfully applied linguistic segmentation with contraction protection and target length {target_length}")
                    temp_file = output_file
                else:
                    logger.warning("Linguistic segmentation failed, continuing with previous file")
            else:
                logger.debug("Skipping linguistic segmentation (not selected)")
            
            # 4. Split at punctuation (periods, question marks, etc.) - Enhanced with contraction awareness
            if self.split_punct_var.get():
                logger.info("Splitting at punctuation (with contraction protection)")
                output_file = f"{base_file}_split_punct.srt"
                self.split_subtitles_at_punctuation_protected(temp_file, output_file)
                temp_file = output_file
            else:
                logger.debug("Skipping splitting at punctuation (not selected)")
            
            # 5. Split at commas (now includes the newly added commas)
            if self.split_commas_var.get():
                logger.info("Splitting at commas")
                output_file = f"{base_file}_split_commas.srt"
                self.split_subtitles_at_commas(temp_file, output_file)
                temp_file = output_file
            else:
                logger.debug("Skipping splitting at commas (not selected)")
            
            # 5.1. NEW: Fix small trailing subtitles for 3rd track (AFTER comma splitting)
            logger.info("Fixing small trailing subtitles for 3rd track placement")
            output_file = f"{base_file}_track3_fixed.srt"
            self.fix_small_trailing_subtitles(temp_file, output_file)
            temp_file = output_file
            
            # 5.2. NEW: Create single subtitle overlaps (ENHANCED FEATURE)
            logger.info("Creating single subtitle overlaps for better flow")
            output_file = f"{base_file}_single_overlaps.srt"
            self.create_single_subtitle_overlaps(temp_file, output_file)
            temp_file = output_file
            
            # 5.3. NEW: Split long track-marked subtitles while preserving markers
            logger.info("Splitting long track-marked subtitles while preserving track assignments")
            output_file = f"{base_file}_track_marked_split.srt"
            self.split_long_track_marked_subtitles(temp_file, output_file)
            temp_file = output_file
            
            # 5.5. Fix cross-subtitle interjection overlaps (EXISTING)
            logger.info("Fixing cross-subtitle interjection overlaps")
            output_file = f"{base_file}_fixed_overlaps.srt"
            self.fix_interjection_overlaps(temp_file, output_file)
            temp_file = output_file
            
            # 6. Split any remaining long subtitles as a final fallback
            if self.split_long_var.get():
                logger.info("Splitting remaining long subtitles")
                output_file = f"{base_file}_split_long.srt"
                self.split_long_subtitles(temp_file, output_file, max_length=40)
                temp_file = output_file
            else:
                logger.debug("Skipping splitting long subtitles (not selected)")
            
            # 7. Add counters for duplicate words (if enabled)
            if self.add_duplicate_counters_var.get():
                logger.info("Adding counters for consecutive duplicate words")
                output_file = f"{base_file}_duplicates_numbered.srt"
                
                success = self.add_counters_for_duplicates(temp_file, output_file)
                if success:
                    logger.info("Successfully added counters for consecutive duplicate words")
                    temp_file = output_file
                else:
                    logger.warning("Failed to add duplicate counters, continuing with previous file")
            else:
                logger.debug("Skipping adding counters for duplicate words (not selected)")
                
            # 8. Always clean the final file
            logger.info("Cleaning final processed file")
            self.processed_srt_path = f"{base_file}_processed.srt"
            self.clean_srt_file(temp_file, self.processed_srt_path)
            
            # Clean up temporary files
            temp_files = [
                f"{base_file}_temp.srt",
                f"{base_file}_comma_replaced.srt",
                f"{base_file}_first_commas_replaced.srt",
                f"{base_file}_interjection_punctuation.srt",
                f"{base_file}_added_commas_before.srt",
                f"{base_file}_added_commas_after.srt",
                f"{base_file}_linguistic_segmented.srt",
                f"{base_file}_split_punct.srt",
                f"{base_file}_split_commas.srt",
                f"{base_file}_track3_fixed.srt",
                f"{base_file}_single_overlaps.srt",
                f"{base_file}_track_marked_split.srt",
                f"{base_file}_fixed_overlaps.srt",
                f"{base_file}_split_long.srt",
                f"{base_file}_duplicates_numbered.srt"
            ]
            
            logger.debug("Cleaning up temporary files")
            for file in temp_files:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                        logger.debug(f"Removed temporary file: {file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {file}: {str(e)}")
                        
            self.status_label.config(text="SRT preprocessing complete.")
            logger.info("Enhanced SRT preprocessing complete with contraction protection, single subtitle overlaps and track-marked splitting")
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing SRT file: {str(e)}", exc_info=True)
            self.show_error(f"Error preprocessing SRT file: {str(e)}")
            return False

    def split_subtitles_at_punctuation_protected(self, input_file, output_file):
        """Split subtitles at sentence-ending punctuation marks with contraction protection."""
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Splitting subtitles at punctuation (protected) from {input_file} to {output_file}")
        
        try:
            import pysrt
            
            # Load SRT file
            subs = pysrt.open(input_file, encoding='utf-8')
            logger.debug(f"Loaded {len(subs)} subtitles for protected punctuation splitting")
            
            new_subs = pysrt.SubRipFile()
            counter = 1
            
            # Common contractions to protect
            contractions = [
                "that's", "it's", "he's", "she's", "we're", "you're", "they're",
                "i'm", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't",
                "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
                "haven't", "hasn't", "hadn't", "i've", "you've", "we've", "they've",
                "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "i'd",
                "you'd", "he'd", "she'd", "we'd", "they'd", "there's", "here's",
                "what's", "where's", "when's", "who's", "how's", "why's", "let's"
            ]
            
            # Abbreviations to protect from splitting
            abbreviations = ["Mr.", "Ms.", "Dr.", "Prof.", "Jr.", "Sr.", "Mrs.", "St.", "Co.", "Inc.", "Ltd.", "Gov."]
            
            for sub_index, sub in enumerate(subs):
                logger.debug(f"Processing subtitle {sub_index+1}/{len(subs)}: {sub.text}")
                
                # Skip empty subtitles
                if not sub.text.strip():
                    continue
                
                # Split the subtitle at sentence boundaries with contraction protection
                split_segments = self._split_at_sentence_boundaries_protected(sub.text, abbreviations, contractions)
                logger.debug(f"Split '{sub.text}' into {len(split_segments)} sentences: {split_segments}")
                
                if len(split_segments) <= 1:
                    # No splitting needed
                    new_subs.append(pysrt.SubRipItem(
                        index=counter,
                        start=sub.start,
                        end=sub.end,
                        text=sub.text.strip()
                    ))
                    counter += 1
                else:
                    # Create multiple subtitles with proportional timing
                    start_ms = sub.start.ordinal
                    total_duration = sub.end.ordinal - sub.start.ordinal
                    
                    for i, segment in enumerate(split_segments):
                        # Calculate proportional timing
                        if i == len(split_segments) - 1:
                            # Last segment gets remaining time
                            segment_end = sub.end.ordinal
                        else:
                            # Proportional duration based on character count
                            total_chars = sum(len(seg) for seg in split_segments)
                            segment_chars = len(segment)
                            proportion = segment_chars / total_chars if total_chars > 0 else 1/len(split_segments)
                            segment_duration = max(300, int(proportion * total_duration))  # Minimum 300ms
                            segment_end = start_ms + segment_duration
                        
                        # Ensure we don't exceed original end time
                        segment_end = min(segment_end, sub.end.ordinal)
                        
                        new_subs.append(pysrt.SubRipItem(
                            index=counter,
                            start=pysrt.SubRipTime.from_ordinal(start_ms),
                            end=pysrt.SubRipTime.from_ordinal(segment_end),
                            text=segment.strip()
                        ))
                        counter += 1
                        start_ms = segment_end
            
            # Save the new subtitles
            new_subs.save(output_file, encoding='utf-8')
            logger.info(f"Split subtitles at punctuation (protected) into {len(new_subs)} total subtitles")
            return True
            
        except Exception as e:
            logger.error(f"Error splitting at punctuation (protected): {str(e)}")
            logger.exception(e)
            return False

    def _split_at_sentence_boundaries_protected(self, text, abbreviations, contractions):
        """Split text at sentence boundaries while protecting abbreviations and contractions."""
        logger = logging.getLogger('SubtitleGenerator')
        
        # Protect contractions first
        protected_text = text
        contraction_map = {}
        
        for i, contraction in enumerate(contractions):
            placeholder = f"<CONTRACTION_{i}>"
            
            # Case-insensitive replacement but preserve original case
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            
            def replace_func(match):
                original = match.group(0)
                contraction_map[placeholder] = original
                return placeholder
            
            protected_text = pattern.sub(replace_func, protected_text)
        
        # Protect abbreviations
        for abbr in abbreviations:
            protected_text = protected_text.replace(abbr, abbr.replace(".", "<DOT>"))
        
        # Protect ellipses
        protected_text = protected_text.replace("...", "<ELLIPSIS>").replace("..", "<ELLIPSIS>")
        
        # Split at sentence endings, but not within contraction placeholders
        pattern = r'(?<=[.!?])\s+|(?<=[.!?])$|(?<=<ELLIPSIS>)\s+|(?<=<ELLIPSIS>)'
        sentence_endings = re.split(pattern, protected_text)
        sentences = [s.strip() for s in sentence_endings if s.strip()]
        
        # Restore ellipses and dots
        sentences = [s.replace("<ELLIPSIS>", "...").replace("<DOT>", ".") for s in sentences]
        
        # Restore contractions
        restored_sentences = []
        for sentence in sentences:
            restored_sentence = sentence
            for placeholder, original in contraction_map.items():
                restored_sentence = restored_sentence.replace(placeholder, original)
            restored_sentences.append(restored_sentence)
        
        logger.debug(f"Split '{text}' into {len(restored_sentences)} sentences with contraction protection: {restored_sentences}")
        return restored_sentences

    def _is_interjection_candidate(self, text, interjection_words):
        """
        Check if subtitle text is a candidate for creating overlaps with following content.
        
        Criteria:
        1. Short text (1-4 words)
        2. Contains known interjection words
        3. Often ends with comma (but not required)
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        # Clean and analyze the text
        clean_text = text.strip()
        words = clean_text.split()
        word_count = len(words)
        
        logger.debug(f"    Interjection analysis: '{clean_text}' -> {word_count} words")
        
        # Must be short (1-4 words)
        if word_count < 1 or word_count > 4:
            logger.debug(f"    Not interjection: word count {word_count} outside range 1-4")
            return False
        
        # Check if any word is an interjection
        contains_interjection = False
        for word in words:
            clean_word = word.lower().strip('.,!?;:')
            if clean_word in interjection_words:
                contains_interjection = True
                logger.debug(f"    Found interjection word: '{clean_word}'")
                break
        
        if not contains_interjection:
            logger.debug(f"    Not interjection: no interjection words found")
            return False
        
        # Additional scoring based on characteristics
        score = 0
        
        # Boost score for comma endings (indicates continuation)
        if clean_text.endswith(','):
            score += 3
            logger.debug(f"    +3 for comma ending")
        
        # Boost score for single words
        if word_count == 1:
            score += 2
            logger.debug(f"    +2 for single word")
        
        # Boost score for very short phrases
        if word_count <= 2:
            score += 1
            logger.debug(f"    +1 for very short ({word_count} words)")
        
        # Boost score for common interjection patterns
        first_word = words[0].lower().strip('.,!?;:')
        if first_word in ['oh', 'well', 'but', 'and', 'so']:
            score += 2
            logger.debug(f"    +2 for high-priority interjection '{first_word}'")
        
        # Require minimum score to qualify
        min_score = 2
        is_candidate = score >= min_score
        
        logger.debug(f"    Final score: {score}, candidate: {is_candidate}")
        return is_candidate

    def _is_overlap_worthy_content(self, text, worthy_words):
        """
        Check if subtitle content would benefit from overlapping with a preceding interjection.
        
        Criteria:
        1. Short content (1-6 words)
        2. Starts with pronouns, articles, names, or continuation words
        3. Not another interjection itself
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        # Clean and analyze the text
        clean_text = text.strip()
        words = clean_text.split()
        word_count = len(words)
        
        logger.debug(f"    Overlap-worthy analysis: '{clean_text}' -> {word_count} words")
        
        # Should be reasonably short to benefit from overlap
        if word_count < 1 or word_count > 6:
            logger.debug(f"    Not overlap-worthy: word count {word_count} outside range 1-6")
            return False
        
        # Get first word for analysis
        first_word = words[0].lower().strip('.,!?;:')
        logger.debug(f"    First word: '{first_word}'")
        
        # Check if first word makes this suitable for overlap
        if first_word in worthy_words:
            logger.debug(f"    Overlap-worthy: starts with '{first_word}'")
            return True
        
        # Additional patterns that benefit from overlap
        
        # Short phrases starting with common words
        if word_count <= 3 and first_word in ['like', 'just', 'really', 'very', 'so', 'too', 'also']:
            logger.debug(f"    Overlap-worthy: short phrase starting with '{first_word}'")
            return True
        
        # Names or proper nouns (capitalized words)
        if words[0][0].isupper() and not clean_text.endswith('?'):  # Not questions
            logger.debug(f"    Overlap-worthy: starts with capitalized word '{words[0]}'")
            return True
        
        # Numbers or short measurements
        if first_word.isdigit() or first_word in ['one', 'two', 'three', 'four', 'five']:
            logger.debug(f"    Overlap-worthy: starts with number '{first_word}'")
            return True
        
        logger.debug(f"    Not overlap-worthy: doesn't match criteria")
        return False

    def _calculate_ideal_overlap(self, interjection_text, following_text):
        """
        Calculate the ideal overlap duration in milliseconds based on content characteristics.
        
        Returns:
        - Overlap duration in milliseconds
        - 0 if no overlap recommended
        """
        logger = logging.getLogger('SubtitleGenerator')
        
        interjection_words = len(interjection_text.split())
        following_words = len(following_text.split())
        
        logger.debug(f"    Calculating overlap: '{interjection_text}' ({interjection_words}w) + '{following_text}' ({following_words}w)")
        
        # Base overlap duration
        base_overlap = 400  # 400ms base
        
        # Adjust based on interjection characteristics
        if interjection_words == 1:
            # Single word interjections can have longer overlaps
            base_overlap += 200
            logger.debug(f"    +200ms for single word interjection")
        
        if interjection_text.strip().endswith(','):
            # Comma indicates strong continuation
            base_overlap += 150
            logger.debug(f"    +150ms for comma continuation")
        
        # Adjust based on following content
        if following_words <= 2:
            # Short following content benefits from longer overlap
            base_overlap += 100
            logger.debug(f"    +100ms for short following content")
        elif following_words >= 5:
            # Longer content needs less overlap
            base_overlap -= 100
            logger.debug(f"    -100ms for longer following content")
        
        # Check for specific high-value patterns
        interjection_first = interjection_text.split()[0].lower().strip('.,!?;:')
        following_first = following_text.split()[0].lower().strip('.,!?;:')
        
        high_value_patterns = [
            ('oh', ['a', 'the', 'jesus', 'god', 'christ', 'man', 'guy']),
            ('well', ['i', 'you', 'that', 'this', 'there']),
            ('but', ['i', 'you', 'that', 'wait', 'no']),
            ('and', ['i', 'you', 'then', 'now', 'so']),
            ('yeah', ['i', 'you', 'that', 'right', 'sure'])
        ]
        
        for interjection_word, good_followers in high_value_patterns:
            if interjection_first == interjection_word and following_first in good_followers:
                base_overlap += 200
                logger.debug(f"    +200ms for high-value pattern: '{interjection_word}' + '{following_first}'")
                break
        
        # Ensure reasonable bounds
        min_overlap = 300
        max_overlap = 800
        
        final_overlap = max(min_overlap, min(base_overlap, max_overlap))
        
        logger.debug(f"    Final overlap duration: {final_overlap}ms")
        return final_overlap

    def split_subtitles_at_punctuation(self, input_file, output_file):
        """Split subtitles at sentence-ending punctuation marks."""
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Splitting subtitles at punctuation from {input_file} to {output_file}")
        
        try:
            import pysrt
            
            # Load SRT file
            subs = pysrt.open(input_file, encoding='utf-8')
            logger.debug(f"Loaded {len(subs)} subtitles for punctuation splitting")
            
            new_subs = pysrt.SubRipFile()
            counter = 1
            
            # Abbreviations to protect from splitting
            abbreviations = ["Mr.", "Ms.", "Dr.", "Prof.", "Jr.", "Sr.", "Mrs.", "St.", "Co.", "Inc.", "Ltd.", "Gov."]
            
            for sub_index, sub in enumerate(subs):
                logger.debug(f"Processing subtitle {sub_index+1}/{len(subs)}: {sub.text}")
                
                # Skip empty subtitles
                if not sub.text.strip():
                    continue
                
                # Split the subtitle at sentence boundaries
                split_segments = self._split_at_sentence_boundaries(sub.text, abbreviations)
                logger.debug(f"Split '{sub.text}' into {len(split_segments)} sentences: {split_segments}")
                
                if len(split_segments) <= 1:
                    # No splitting needed
                    new_subs.append(pysrt.SubRipItem(
                        index=counter,
                        start=sub.start,
                        end=sub.end,
                        text=sub.text.strip()
                    ))
                    counter += 1
                else:
                    # Create multiple subtitles with proportional timing
                    start_ms = sub.start.ordinal
                    total_duration = sub.end.ordinal - sub.start.ordinal
                    
                    for i, segment in enumerate(split_segments):
                        # Calculate proportional timing
                        if i == len(split_segments) - 1:
                            # Last segment gets remaining time
                            segment_end = sub.end.ordinal
                        else:
                            # Proportional duration based on character count
                            total_chars = sum(len(seg) for seg in split_segments)
                            segment_chars = len(segment)
                            proportion = segment_chars / total_chars if total_chars > 0 else 1/len(split_segments)
                            segment_duration = max(300, int(proportion * total_duration))  # Minimum 300ms
                            segment_end = start_ms + segment_duration
                        
                        # Ensure we don't exceed original end time
                        segment_end = min(segment_end, sub.end.ordinal)
                        
                        new_subs.append(pysrt.SubRipItem(
                            index=counter,
                            start=pysrt.SubRipTime.from_ordinal(start_ms),
                            end=pysrt.SubRipTime.from_ordinal(segment_end),
                            text=segment.strip()
                        ))
                        counter += 1
                        start_ms = segment_end
            
            # Save the new subtitles
            new_subs.save(output_file, encoding='utf-8')
            logger.info(f"Split subtitles at punctuation into {len(new_subs)} total subtitles")
            return True
            
        except Exception as e:
            logger.error(f"Error splitting at punctuation: {str(e)}")
            logger.exception(e)
            return False

    def _split_at_sentence_boundaries(self, text, abbreviations):
        """Split text at sentence boundaries while protecting abbreviations."""
        logger = logging.getLogger('SubtitleGenerator')
        
        # Protect abbreviations
        protected_text = text
        for abbr in abbreviations:
            protected_text = protected_text.replace(abbr, abbr.replace(".", "<DOT>"))
        
        # Protect ellipses
        protected_text = protected_text.replace("...", "<ELLIPSIS>").replace("..", "<ELLIPSIS>")
        
        # Split at sentence endings
        pattern = r'(?<=[.!?])\s+|(?<=[.!?])$|(?<=<ELLIPSIS>)\s+|(?<=<ELLIPSIS>)'
        sentence_endings = re.split(pattern, protected_text)
        sentences = [s.strip() for s in sentence_endings if s.strip()]
        
        # Restore ellipses and dots
        sentences = [s.replace("<ELLIPSIS>", "...").replace("<DOT>", ".") for s in sentences]
        
        logger.debug(f"Split '{text}' into {len(sentences)} sentences: {sentences}")
        return sentences

    def add_counters_for_duplicates(self, input_file, output_file):
        """Add counters (X2, X3, etc.) for consecutive repeated FULL TEXT (not just words)."""
        logger = logging.getLogger('SubtitleGenerator')
        logger.info(f"Adding counters for consecutive duplicate FULL TEXT from {input_file} to {output_file}")
        
        try:
            # Parse SRT file
            subtitles = self._parse_srt_file(input_file)
            logger.debug(f"Loaded {len(subtitles)} subtitles")
            
            # Track consecutive text sequences
            current_sequence = []  # List of (normalized_text, subtitle_index)
            
            # Process each subtitle
            for i, subtitle in enumerate(subtitles):
                content = " ".join(subtitle["content"])
                original_content = content
                
                logger.debug(f"Processing subtitle {i+1}/{len(subtitles)}: '{content}'")
                
                # Normalize the ENTIRE text for comparison (remove punctuation, lowercase, strip)
                normalized_text = self._normalize_text_for_comparison(content)
                logger.debug(f"Normalized text: '{normalized_text}' from '{content}'")
                
                if normalized_text:
                    # Check if this text continues the current sequence
                    if current_sequence and current_sequence[-1][0] == normalized_text:
                        # This text continues the sequence - EXACT same text as previous
                        current_sequence.append((normalized_text, i))
                        sequence_count = len(current_sequence)
                        
                        logger.info(f"Found consecutive identical text '{normalized_text}' (occurrence #{sequence_count})")
                        logger.debug(f"  Current sequence length: {sequence_count}")
                        
                        # Add counter for the second occurrence and beyond
                        if sequence_count >= 2:
                            counter_suffix = f"X{sequence_count}"
                            modified_content = self._add_counter_to_content(content, counter_suffix)
                            
                            logger.debug(f"  Adding counter to subtitle {i + 1}")
                            logger.debug(f"  BEFORE: '{original_content}'")
                            logger.debug(f"  AFTER:  '{modified_content}'")
                            
                            # Update the subtitle content
                            subtitle["content"] = [modified_content]
                    else:
                        # This text breaks the sequence or starts a new one
                        if current_sequence:
                            prev_text = current_sequence[-1][0]
                            logger.debug(f"Sequence broken. Previous text: '{prev_text}', New text: '{normalized_text}'")
                        
                        # Start new sequence with this text
                        current_sequence = [(normalized_text, i)]
                        logger.debug(f"Started new sequence with '{normalized_text}' at subtitle {i + 1}")
                else:
                    # No meaningful text found, break any current sequence
                    if current_sequence:
                        logger.debug(f"Sequence broken due to empty/invalid content.")
                        current_sequence = []
                    logger.debug(f"Could not normalize text from '{content}', sequence reset")
            
            # Write the modified SRT file
            self._write_srt_file(subtitles, output_file)
            logger.info(f"Successfully processed {len(subtitles)} subtitles, adding counters for consecutive identical text")
            return True
            
        except Exception as e:
            logger.error(f"Error adding counters for duplicates: {str(e)}")
            logger.exception(e)
            return False

    def _normalize_text_for_comparison(self, content):
        """Normalize the ENTIRE text for comparison by removing punctuation, extra spaces, and converting to lowercase."""
        # Remove all punctuation and convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', content.strip().lower())
        # Replace multiple spaces with single space and strip
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        logger = logging.getLogger('SubtitleGenerator')
        logger.debug(f"Normalized '{content}' -> '{normalized}'")
        
        return normalized if normalized else None

    def _add_counter_to_content(self, content, counter_suffix):
        """Add counter suffix to content while preserving punctuation."""
        # Strategy: Find the last word and add the counter before any trailing punctuation
        # Examples:
        # "okay," -> "okayX2,"
        # "Jesus" -> "JesusX2"
        # "oh." -> "ohX2."
        
        # Pattern to match the content with optional trailing punctuation
        pattern = r'^(.*?)([^\w\s]*)(\s*)$'
        match = re.match(pattern, content)
        
        if match:
            main_content, punctuation, trailing_space = match.groups()
            
            # Find the last word in the main content
            words = main_content.strip().split()
            if words:
                # Replace the last word with word + counter
                words[-1] = words[-1] + counter_suffix
                modified_main = ' '.join(words)
                return f"{modified_main}{punctuation}{trailing_space}"
        
        # Fallback: just append the counter to the trimmed content
        return f"{content.strip()}{counter_suffix}"

    def _parse_srt_file(self, input_file):
        """Parse an SRT file into a list of subtitle dictionaries."""
        logger = logging.getLogger('SubtitleGenerator')
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        subtitles = []
        current_subtitle = {"index": None, "timestamps": None, "content": []}
        
        for line in lines:
            line = line.strip()
            
            if not line:  # Empty line indicates end of a subtitle
                if current_subtitle["index"] is not None:
                    subtitles.append(current_subtitle)
                    current_subtitle = {"index": None, "timestamps": None, "content": []}
                continue
            
            if current_subtitle["index"] is None:
                # This is the subtitle index
                current_subtitle["index"] = line
            elif current_subtitle["timestamps"] is None:
                # This is the timestamp line
                current_subtitle["timestamps"] = line
            else:
                # This is content
                current_subtitle["content"].append(line)
        
        # Don't forget the last subtitle if file doesn't end with an empty line
        if current_subtitle["index"] is not None:
            subtitles.append(current_subtitle)
        
        return subtitles

    def _write_srt_file(self, subtitles, output_file):
        """Write subtitle dictionaries to an SRT file."""
        logger = logging.getLogger('SubtitleGenerator')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, subtitle in enumerate(subtitles):
                # Write index
                f.write(f"{subtitle['index']}\n")
                
                # Write timestamps
                f.write(f"{subtitle['timestamps']}\n")
                
                # Write content
                for line in subtitle["content"]:
                    f.write(f"{line}\n")
                
                # Add blank line between subtitles (except after the last one)
                if i < len(subtitles) - 1:
                    f.write("\n")

def main():
    logger.info("===== Starting Subtitle Generator Application =====")
    try:
        logger.debug("Creating main window")
        root = tk.Tk()
        logger.debug("Initializing SubtitleGenerator")
        app = SubtitleGenerator(root)
        logger.info("Application initialized, starting main loop")
        root.mainloop()
        logger.info("Application closed normally")
    except Exception as e:
        logger.critical(f"Critical error starting application: {str(e)}", exc_info=True)
        print(f"Error starting application: {str(e)}")

if __name__ == "__main__":
    try:
        logger.info("Script executed directly, starting main function")
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {str(e)}", exc_info=True)
        print(f"Error starting application: {str(e)}")