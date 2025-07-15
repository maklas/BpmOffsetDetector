import librosa
import numpy as np
import math

# Author: Maklas

# These parameters MUST NOT be changed. The whole algorithm was fine-tuned on them.
# I mean I don't claim these are the best for your use case. But if you change these,
# you better have a large database of samples to test it on.
# Also, if you change this you'll probably have to fine-tune other magic numbers across this file
__SR__ = 44100
__HOP__ = 512

# Can be set to either 1 or 0.5. Increasing it more is just dumb.
# Decreasing increases error rate and time it takes to detect
__BPM_PRECISION__ = 0.5
# Max BPM. Everything beyond will be forcefully halved
__BPM_LIMIT__ = 200
# Warning! Linearly increases BPM detection time.
# Reduce -> accuracy and time it takes to detect falls
# Increase -> Accuracy doesn't really grow all that much, but time to detect will grow
__MATCH_SCORE_CHECKS__ = 16

def _get_onsets_from_strength(onset_strength, sr):
    """
    This is the only suitable configuration for the most correct BPM and offset detection
    """
    return librosa.onset.onset_detect(sr=sr, onset_envelope=onset_strength, hop_length=__HOP__, units='frames')

def _get_gaps(onsets):
    """
    Gaps between onsets in frames. But we wrap it from 8 to 14 frames.
    Which means we're initially looking in range from 90 to 172 BPM.
    """
    gaps = []
    min_gap = 7
    max_gap = min_gap * 2
    for i in range(1, len(onsets)):
        distance = onsets[i] - onsets[i - 1]
        while distance > 0 and distance < min_gap:
            distance *= 2
        while distance > max_gap:
            distance = round(distance / 2)
        if distance <= min_gap:
            distance *= 2
        gaps.append(distance)
    return gaps

def _get_ballpark_bpms(gaps):
    """
    Using extracted gaps we're trying to narrow down the BPM range that we're going to brute force
    """
    window = 5
    assert (window - 1) % 2 == 0
    wings = int((window - 1) / 2)
    max = np.max(gaps)
    bins = [0 for _ in range(max + 1)]

    for g in gaps:
        bins[g] += 1
    for i in range(max - window + 2, max, 2):
        bins[int(i / 2)] = bins[i] + bins[i + 1]

    center_index = 0
    center_max = 0
    for i in range(wings, len(bins) - wings):
        sum = 0
        for j in range(i - wings, i + wings + 1):
            sum += bins[j] / (abs(i - j) + 1)
        if sum > center_max:
            center_index = i
            center_max = sum

    min_frames = center_index - wings
    max_frames = center_index + wings

    min_bpm = 60 / (((max_frames * __HOP__) / __SR__) * 4)
    max_bpm = 60 / (((min_frames * __HOP__) / __SR__) * 4)


    bpms = []
    if max_bpm > __BPM_LIMIT__:
        bpms += [float(_) for _ in np.arange(math.floor(min_bpm), __BPM_LIMIT__, __BPM_PRECISION__)]
        bpms += [float(_) for _ in np.arange(math.floor(__BPM_LIMIT__ / 2), math.ceil(max_bpm / 2), __BPM_PRECISION__)]
        bpms = sorted(list(set(bpms)))
    else:
        bpms = [float(_) for _ in np.arange(math.floor(min_bpm), math.ceil(max_bpm), __BPM_PRECISION__)]
    return bpms


def _get_scores(onsets, bpm):
    return [_get_match_score(onsets, bpm), _get_avg_change_score(onsets, bpm)]

def _get_match_score(onsets, bpm):
    """
    Score of how well this bpm matching the onsets.
    A number of fitting onsets for this BPM
    """
    times = librosa.frames_to_time(onsets, sr=__SR__, hop_length=__HOP__)
    step = 60 / bpm
    error_window = step / __MATCH_SCORE_CHECKS__
    results = []
    for i in np.arange(0, step, step / __MATCH_SCORE_CHECKS__):
        r = 0
        for t in times:
            d = (t - i) % step
            if d > step / 2:
                d -= step
            if abs(d) < error_window:
                r += 1
        results.append(r)

    return max(results)

def _get_avg_change_score(onsets, bpm):
    """
    Score of how well this bpm matching the onsets.
    The average change of distances to the offsets.
    """
    times = librosa.frames_to_time(onsets, sr=__SR__, hop_length=__HOP__)
    step = 60 / bpm
    distances = []
    for t in times:
        d = t % step
        if d > step / 2:
            d -= step
        distances.append(abs(d))
    distances2 = []
    for i in range(0, len(distances) - 1):
        dd = abs(distances[i + 1] - distances[i])
        distances2.append(dd)
    return np.average(distances2) * bpm

def _select_best_index(scores):
    """
    We're looking for a value that has the most change in both scores across neighbouring BPMs
    """
    scores = np.array(scores)
    match_scores = scores[:, 0]
    avg_scores = scores[:, 1]

    best_std_indexes = list(range(len(avg_scores)))

    best = {}
    for i in best_std_indexes:
        main_val = avg_scores[i]
        surrounding_values = []
        #we get 2 from the left and 2 from the right of the value
        for j in range(max(0, i - 2), i):
            surrounding_values.append(avg_scores[j])
        for j in range(i + 1, min(len(avg_scores), i + 3)):
            surrounding_values.append(avg_scores[j])

        change = main_val / np.average(surrounding_values)
        best[str(i)] = change if change > 1 else 1 / change

    for i in best_std_indexes:
        main_val = match_scores[i]
        surrounding_values = []
        #we get 2 from the left and 2 from the right of the value
        for j in range(max(0, i - 2), i):
            surrounding_values.append(match_scores[j])
        for j in range(i + 1, min(len(match_scores), i + 3)):
            surrounding_values.append(match_scores[j])

        change = main_val / np.average(surrounding_values)
        best[str(i)] *= change if change > 1 else 1 / change

    return int(max(best, key=best.get))

def _find_time_scale():
    # TODO: I'm sorry, I'm too lazy for now
    return 4

def _find_offset(onset_strength, bpm, sr, x=1.25):
    """
    This algorithm is not that bad, but honestly I think it could be better.
    Spectrogram shows offset quite clearly, and may be that's the key.
    But here we do magic numbers for some reason
    """
    dt = (60 / bpm)
    timings = librosa.onset.onset_detect(sr=sr, onset_envelope=onset_strength, hop_length=__HOP__, units='time')
    error_room = __HOP__/44100 # +- 1 frame. TODO: test different values

    timing_count = {}
    timing_map = {}
    for t in np.arange(0, dt, dt / 128): #128 - is precision. May be lowering it won't do harm. Never tested. 128 is fine. TODO: needs testing
        count = 0
        matched_timings = []
        for i in timings:
            delta = (i - t) % dt
            delta = delta - dt if delta > dt / 2 else delta
            if abs(delta) < error_room:
                count += 1
                matched_timings.append(delta)
        timing_count[t] = count
        timing_map[t] = matched_timings

    best = max(timing_count, key=timing_count.get) # best is the one with the most onsets
    best += np.average(timing_map[best]) # even more precise position by averaging all matched onsets
    best = best % dt # Skip a beat if required
    best = best - ((__HOP__/sr) * x) # subtract 1.25 pixels (hops) because onset detection is usually lagging behind a bit. Value of 1.25 was figured out by testing many songs
    offset = best - dt if best > dt/2 else best # look for closest value. Closest might be on the negative side
    add = dt - offset if offset >= 0 else abs(offset)
    return offset, add


def load(audio_file):
    """
    Use this loader if you need length information about a song before detecting BPM
    """
    y, sr = librosa.load(audio_file, sr=__SR__)
    return y, sr

def detect(audio_path=None, y=None, sr=None, bpm=None, detect_offset=True, detect_time_scale=True):
    """
    :param audio_path: Path to an audio file
    :param y: audio time series from librosa.load(). Either supply audio file path or y + sr
    :param sr: sample rate
    :param bpm: You can specify BPM if it's known
    :param detect_offset: boolean. If True, the function will also return OFFSET and ADD (takes additional time to detect)
    :param detect_time_scale: boolean. If True, the function will also return TIME_SCALE (takes additional time to detect)
    :return: BPM of the song,
            TIME_SCALE (Either 3 or 4)
            OFFSET (positive or negative value within +-(60/bpm) / 2) and
            ADD (always positive number of how many seconds of silence to add to audio file to make song offset be = 0)

    Examples
    --------
    >>> bpm, ts, offset, add = bpm_offset_detector.detect('my_song.ogg')
    >>> bpm
    125
    >>> offset
    0.123
    >>> add
    0.357
    >>> ...append 0.357 seconds of silence
    """
    if audio_path is None and (y is None or sr is None):
        raise AttributeError('Specify either audio_path or y + sr')
    if bpm is not None and not detect_offset:
        raise AttributeError('BPM is specified and offset is not needed. Why are you calling me?')
    if y is None or sr is None:
        y, sr = load(audio_path)
    elif sr != __SR__:
        raise AttributeError('BPM/offset detection expects only sample rate of 44100')
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.average)
    if bpm is None:
        onsets = _get_onsets_from_strength(onset_strength, sr)
        gaps = _get_gaps(onsets)
        bpms = _get_ballpark_bpms(gaps)
        scores = [_get_scores(onsets, b) for b in bpms]
        best_bpm_index = _select_best_index(scores)
        bpm = bpms[best_bpm_index]

    if not detect_offset:
        if detect_time_scale:
            return bpm, _find_time_scale()
        else:
            return bpm

    offset, add = _find_offset(onset_strength, bpm, sr)
    if detect_time_scale:
        return bpm, _find_time_scale(), offset, add
    else:
        return bpm, offset, add