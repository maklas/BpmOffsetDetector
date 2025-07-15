# BpmOffsetDetector

BpmOffsetDetector is a Python script designed to analyze and detect BPM (beats per minute) and offsets for music.

## Features
- Detects a BPM for a static BPM songs where BPM is a whole number or is at least divisible by 0.5
- Detects an offset for a song to know where the beats fall 
- Simple and easy-to-use Python script

## Limits
- It will most likely fail if bpm changes throughout the song.
- It assumes song's BPM is a whole number, which is fine for most of the time, but not always.  

## Usage
1. Make sure ffmpeg is installed on your pc
2. You'll need a librosa library as a dependency
3. Call it when you need to find out BPM for a song and/or song offset:
```python
import bpm_offset_detector

bpm, time_scale, offset, add = bpm_offset_detector.detect('song.ogg')
print(bpm) # 125
print(offset) # 0.123 
print(add) # 0.357
#...append 0.357 seconds of silence to make the song's first beat start at exactly 0.0 seconds
```


## Some stats
![Correct guess statistics](test/Stats.png)
Out of 1719 songs tested:
- 1653 are 4/4 timescale and 66 are 3/4
- only 1568 songs are considered to be detectable (meaning their bpm is divisible by 0.5)
- Out of them, 1534 were detected correctly, implying 97.83% success rate for detectable and 89.23% success rate overall

## Author
- Maklas
