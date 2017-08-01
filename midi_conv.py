import midi
import numpy as np
import sys


def midi2numpy(midi_path, filename, track=1):
    """
    Tones of 1 track of a midi file will be extracted (without duration)
    into a numpy array.

    :param midi_path: path of a midi file
    :param filename: filename for numpy output
    """
    # exctract pattern from midi file (1 track)
    pattern = midi.read_midifile(midi_path)

    # this will be filled for returning
    a = np.empty(shape=(0, 1))

    for p in pattern[track]:
        if type(p) == midi.events.NoteOnEvent:
            b = np.array([[p.data[0]]])
            a = np.concatenate((a, b), axis=0)

    np.savetxt(filename + ".np", a)


def numpy2midi(numpy_path, filename, dur=100, velo=50, inst=1):
    """
    A numpy file that represents tones for a melody gets converted into a 
    midi file
    :param numpy_path: path of the numpy file
    :param filename: filename for midi output
    :param dur: duration of every tone (equidistant)
    :param velo: velocity of every tone
    :param inst: instrument (see:
    https://en.wikipedia.org/wiki/General_MIDI#Program_change_events)
    """

    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    data = np.loadtxt(numpy_path, dtype=np.int16)

    track.append(midi.ProgramChangeEvent(tick=0, channel=0, data=[inst]))

    # make equidistant tones with fixed velocity
    for d in data:
        track.append(midi.NoteOnEvent(tick=0, channel=0, data=[d, velo]))
        track.append(midi.NoteOffEvent(tick=dur, channel=0, data=[d, velo]))
    track.append(midi.EndOfTrackEvent(tick=1))

    midi.write_midifile(filename + ".mid", pattern)


if __name__ == "__main__":
    # choose midi file
    parms = sys.argv[1:]
    if len(parms) > 1:
        file = parms[0]
        destination = parms[1]
    else:
        print("No file chosen!")
        sys.exit(1)

    if file[-3:] == "mid":
        midi2numpy(file, destination)
    elif file[-2:] == "np":
        numpy2midi(file, destination)
