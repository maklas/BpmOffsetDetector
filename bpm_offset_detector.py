import librosa
import numpy as np

# Author: Maklas

# These parameters MUST NOT be changed. The whole algorithm was fine-tuned on them.
# I mean... I don't claim these are the best for your use case. But if you change these,
# you better have a large database of samples to test it on.
# Also, if you change this, you'll probably have to fine-tune other magic numbers across this file
__SR__ = 44100
__HOP__ = 512

# Can be set to either 1 or 0.5. Increasing it more is just dumb.
# Decreasing increases error rate and time it takes to detect
__BPM_PRECISION__ = 0.5
# Minimal BPM. Everything below that value will be forcefully doubled
__MIN_BPM__ = 90
# Warning! Linearly increases BPM detection time.
# Reduce -> accuracy and time it takes to detect falls
# Increase -> Accuracy doesn't really grow all that much, but time to detect will grow
__MATCH_SCORE_CHECKS__ = 16

def _get_onsets_from_strength(onset_strength, sr):
    """
    This is the only suitable configuration for the most correct BPM and offset detection
    """
    return librosa.onset.onset_detect(sr=sr, onset_envelope=onset_strength, hop_length=__HOP__, units='frames')

def _find_offset(timings, bpm, sr, x=1.25):
    """
    This algorithm is not that bad, but honestly I think it could be better.
    Spectrogram shows offset quite clearly, and may be that's the key.
    But here we do magic numbers for some reason
    """
    dt = (60 / bpm)
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

    best = max(timing_count, key=timing_count.get) # the best is the one with the most onsets
    best += np.average(timing_map[best]) # even more precise position by averaging all matched onsets
    best = best % dt # Skip a beat if required
    best = best - ((__HOP__/sr) * x) # Subtract 1.25 pixels (hops) because onset detection is usually lagging behind a bit. The value of 1.25 was figured out by testing many songs
    offset = best - dt if best > dt/2 else best # Look for the closest value. Closest might be on the negative side
    add = dt - offset if offset >= 0 else abs(offset)
    return offset, add

def _calculate_subdivisions(bpm, timings, onsets, onset_strength, offset):
    bins3 = [0, 0, 0]
    bins4 = [0, 0, 0, 0]

    beat_time = 60 / bpm
    error_window = __HOP__ / __SR__
    step = beat_time / 3
    for i in range(len(timings)):
        time = timings[i] - offset
        beat_index = int(time / step)
        d = time % step
        if d > step / 2:
            d -= step
            beat_index += 1
        if abs(d) < error_window:
            o = onsets[i]
            s = onset_strength[o]
            bins3[beat_index % 3] += s


    step = beat_time / 4
    for i in range(len(timings)):
        time = timings[i] - offset
        beat_index = int(time / step)
        d = time % step
        if d > step / 2:
            d -= step
            beat_index += 1
        if abs(d) < error_window:
            o = onsets[i]
            s = onset_strength[o]
            bins4[beat_index % 4] += s

    score_3 = max([np.average([bins3[1], bins3[2]]), 0.1])
    score_4 = bins4[2]
    score_4_sides = np.average([bins4[1], bins4[3]])
    needs_halving_4 = abs(score_4 - score_4_sides) / max([score_4, score_4_sides, 0.1])

    probability_of_3 = score_3 / (score_3 + score_4)
    if probability_of_3 > 0.5:
        return 3, False
    else:
        return 4, needs_halving_4 > 0.75


def _get_avg_change_score(timings, bpm, offset):
    """
    Score of how well this bpm matching the onsets.
    The average change of distances to the offsets.
    """
    step = 60 / bpm
    distances = []
    for t in timings:
        d = (t - offset) % step
        if d > step / 2:
            d -= step
        distances.append(abs(d))
    distances2 = []
    for i in range(0, len(distances) - 1):
        dd = abs(distances[i + 1] - distances[i])
        distances2.append(dd)
    return np.average(distances2) * bpm


def _get_match_score(timings, bpm):
    """
    Score of how well this bpm matching the onsets.
    A number of fitting onsets for this BPM
    """
    step = 60 / bpm
    window = (__HOP__ / __SR__) * 2 # todo optimize value
    substep = window # todo optimize value
    results = []
    for i in np.arange(0, step, substep):
        r = 0
        for t in timings:
            d = (t - i) % step
            if d > step / 2:
                d -= step
            if abs(d) < window:
                r += 1
        results.append(r)

    max_matches = max(results)
    return max_matches * step, substep * results.index(max_matches)


def _calculate_bpm_scores_and_offset(bpms, timings):
    """
    :return:[bpm, offset, matches, change_score, total_score]
    """
    results = []
    for bpm in bpms:
        score, offset = _get_match_score(timings, bpm)
        change_score = _get_avg_change_score(timings, bpm, offset)
        results.append([bpm, offset, score, change_score, 0])

    best_std_indexes = list(range(len(bpms)))
    for i in best_std_indexes:
        main_val = results[i][3]
        surrounding_values = []
        #we get 2 from the left and 2 from the right of the value
        for j in range(max(0, i - 2), i):
            surrounding_values.append(results[j][3])
        for j in range(i + 1, min(len(results), i + 3)):
            surrounding_values.append(results[j][3])

        change = main_val / np.average(surrounding_values)
        results[i][4] = change if change > 1 else 1 / change

    for i in best_std_indexes:
        main_val = results[i][2]
        surrounding_values = []
        #we get 2 from the left and 2 from the right of the value
        for j in range(max(0, i - 2), i):
            surrounding_values.append(results[j][2])
        for j in range(i + 1, min(len(results), i + 3)):
            surrounding_values.append(results[j][2])

        change = main_val / np.average(surrounding_values)
        results[i][4] *= change if change > 1 else 1 / change
    return results


def _find_bpm_and_sd(bpms, onsets, timings, onset_strength):
    scores = np.array(_calculate_bpm_scores_and_offset(bpms, timings))
    threshold = (max(scores[:,4]) + np.average(scores[:,4])) / 2
    filtered_scores = scores[scores[:, 4] > threshold]
    # Sorted, reverse order, up to 3. These are our best BPM candidates
    sorted_scores = filtered_scores[np.argsort((filtered_scores[:, 4]))][::-1][:3]

    for row in sorted_scores:
        sd, needs_halving = _calculate_subdivisions(row[0], timings, onsets, onset_strength, row[1])
        if sd > 0:
            return row[0], sd, row[1]
    return sorted_scores[0, 0], 4, filtered_scores[0, 1]



def load(audio_file):
    """
    Use this loader if you need length information about a song before detecting BPM
    """
    y, sr = librosa.load(audio_file, sr=__SR__)
    return y, sr

def detect(audio_path=None, y=None, sr=None, bpm=None, sd=None, detect_offset=True):
    """
    :param audio_path: Path to an audio file
    :param y: audio time series from librosa.load(). Either supply audio file path or y + sr
    :param sr: sample rate
    :param bpm: You can specify BPM if it's known
    :param sd: You can specify subdivisions if it's known. Usually it's 4 or 3, meaning how many distinct sounds per beat there are
    :param detect_offset: boolean. If True, the function will also return OFFSET and ADD (takes additional time to detect)
    :return: BPM of the song;
            SD (subdivisions. Either 4 or 3, unless specified a different number);
            OFFSET (positive or negative value within +-(60/bpm) / 2);
            ADD (always positive number of how many seconds of silence to add to audio file to make song offset be = 0)

    Examples
    --------
    >>> bpm, sd, offset, add = bpm_offset_detector.detect('my_song.ogg')
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
    onsets = _get_onsets_from_strength(onset_strength, sr)
    timings = librosa.frames_to_time(onsets, sr=__SR__, hop_length=__HOP__)

    if bpm is None:
        bpms = [float(_) for _ in np.arange(__MIN_BPM__, __MIN_BPM__ * 2, 0.5)]
        bpm, sd, offset = _find_bpm_and_sd(bpms, onsets, timings, onset_strength)
    else:
        offset = None

    if sd is None:
        if offset is None:
            _, offset = _get_match_score(timings, bpm)
        sd, _ = _calculate_subdivisions(bpm, timings, onsets, onset_strength, offset)

    if not detect_offset:
        return bpm, sd

    offset, add = _find_offset(timings, bpm, sr) # todo may be can be optimized with offset found when detecting BPM
    return bpm, sd, offset, add