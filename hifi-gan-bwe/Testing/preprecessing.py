from pydub import AudioSegment

soundfile = AudioSegment.from_file("test.mp3")


ten_seconds = 10000

firstTenSec = soundfile[:ten_seconds]

firstTenSec.export("clippedSound.mp3", format="mp3")