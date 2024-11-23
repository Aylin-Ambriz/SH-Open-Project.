import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional

class SoundHorizonProcessor:
    def __init__(self):
        # Get the current directory (data directory)
        self.base_dir = Path(os.getcwd())
        self.input_file = self.base_dir / 'SH Discography.txt'
        self.output_file = self.base_dir / 'SH JSON processed.json'
        
        print(f"Current directory: {self.base_dir}")
        print(f"Looking for input file at: {self.input_file}")
        
        self.data = {
            "albums": {},
            "interviews": [],
            "other_content": []
        }
        self.current_album = None
        self.current_song = None
        
    def process(self):
        """Main processing function that handles file operations"""
        try:
            # Check if input file exists
            if not self.input_file.exists():
                # Try alternative names
                alternatives = [
                    'SH_Discography.txt',
                    'SH Discography Lyrics.txt',
                    'SH_Discography_Lyrics.txt'
                ]
                
                for alt in alternatives:
                    alt_path = self.base_dir / alt
                    if alt_path.exists():
                        self.input_file = alt_path
                        print(f"Found input file as: {self.input_file}")
                        break
                else:
                    raise FileNotFoundError(
                        f"Input file not found at: {self.input_file}\n"
                        f"Current directory contains: {list(self.base_dir.glob('*.txt'))}"
                    )
            
            # Read input file
            print(f"Reading file from: {self.input_file}")
            with open(self.input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Process text content
            print("Processing content...")
            self._process_text(text)
            
            # Save processed data
            print(f"Saving processed data to: {self.output_file}")
            self._save_json()
            
            print("Processing completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            # Print available files in directory
            print("\nAvailable .txt files in current directory:")
            for file in self.base_dir.glob('*.txt'):
                print(f"- {file.name}")
            return False
    
    def _process_text(self, text: str):
        sections = text.split("~~~")
        
        for section in sections:
            if not section.strip():
                continue
            
            if self._is_album_section(section):
                self._process_album_section(section)
            elif self._is_interview_section(section):
                self._process_interview_section(section)
            else:
                self._process_other_content(section)
    
    def _is_album_section(self, text: str) -> bool:
        return "Story CD" in text or "MAXI Single CD" in text
    
    def _is_interview_section(self, text: str) -> bool:
        return "Interview" in text or "Talk" in text
    
    def _process_album_section(self, text: str):
        # Extract album title
        album_match = re.search(r"(\d+(?:th|st|nd|rd))?\s*(?:Story|MAXI Single)\s*CD\s*-\s*(.+?)(?:\n|$)", text)
        if album_match:
            album_number = album_match.group(1)
            album_title = album_match.group(2).strip()
            
            self.current_album = {
                "title": album_title,
                "number": album_number,
                "type": "Story CD" if "Story CD" in text else "MAXI Single CD",
                "songs": []
            }
            
            # Process songs in album
            songs = re.split(r"~(.+?)~", text)
            for i in range(1, len(songs), 2):
                if i + 1 < len(songs):
                    song_title = songs[i].strip()
                    song_lyrics = songs[i + 1].strip()
                    
                    song_data = {
                        "title": song_title,
                        "lyrics": song_lyrics
                    }
                    
                    self.current_album["songs"].append(song_data)
            
            # Add album to data structure
            self.data["albums"][album_title] = self.current_album
    
    def _process_interview_section(self, text: str):
        # Extract interview title and content
        title_match = re.search(r"Interview.+?(?=\n)", text)
        if title_match:
            title = title_match.group(0).strip()
            
            interview_data = {
                "title": title,
                "content": text.strip(),
                "date": self._extract_date(text)
            }
            
            self.data["interviews"].append(interview_data)
    
    def _process_other_content(self, text: str):
        # Process any other content that isn't an album or interview
        if text.strip():
            self.data["other_content"].append({
                "content": text.strip()
            })
    
    def _extract_date(self, text: str) -> Optional[str]:
        date_match = re.search(r"\d{4}\.\d{2}\.\d{2}", text)
        if date_match:
            return date_match.group(0)
        return None
    
    def _save_json(self):
        """Save processed data to JSON file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

def main():
    processor = SoundHorizonProcessor()
    success = processor.process()
    
    if success:
        print("\nProcessing completed successfully!")
        print(f"Input file: {processor.input_file}")
        print(f"Output file: {processor.output_file}")
    else:
        print("\nProcessing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()