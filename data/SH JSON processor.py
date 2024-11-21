import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional

class SoundHorizonProcessor:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
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
                raise FileNotFoundError(f"Input file {self.input_file} not found")
            
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Read input file
            print(f"Reading file: {self.input_file}")
            with open(self.input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Process text content
            print("Processing content...")
            self._process_text(text)
            
            # Save processed data
            output_file = self.output_dir / 'sound_horizon_data.json'
            print(f"Saving to: {output_file}")
            self._save_json(output_file)
            
            print("Processing completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return False
    
    def _process_text(self, text: str):
        """Process the text content into structured data"""
        sections = text.split("~~~")
        
        for section in sections:
            if not section.strip():
                continue
            
            # Check if section is an album
            if self._is_album_section(section):
                self._process_album_section(section)
            # Check if section is an interview
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
    
    def _save_json(self, output_file: Path):
        """Save processed data to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

def main():
    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    input_file = script_dir / 'data' / 'raw' / 'SH_Discography_Lyrics.txt'
    output_dir = script_dir / 'data' / 'processed'
    
    # Create processor and run
    processor = SoundHorizonProcessor(input_file, output_dir)
    success = processor.process()
    
    if success:
        print("\nFile structure:")
        print(f"Input file: {input_file}")
        print(f"Output directory: {output_dir}")
        print(f"Output file: {output_dir / 'sound_horizon_data.json'}")
    else:
        print("\nProcessing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()