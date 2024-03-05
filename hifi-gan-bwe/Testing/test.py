import audioread
import numpy as np
import torch
from IPython.display import Audio
from hifi_gan_bwe import BandwidthExtender

#Load the pretrained model
model = BandwidthExtender.from_pretrained("hifi-gan-bwe-10-42890e3-vctk-48kHz")

#File to extend
path = 'test.wav'

with audioread.audio_open(path) as input_:
    sample_rate = input_.samplerate
    x = (
        np.hstack([np.frombuffer(b, dtype=np.int16) for b in input_])
        .reshape([-1, input_.channels])
        .astype(np.float32)
        / 32767.0
    )

Audio(x.T, rate=sample_rate, autoplay=False)

with torch.no_grad():
    y = np.stack([model(torch.from_numpy(x), sample_rate) for x in x.T]).T


#Run the model
Audio(y.T, rate=int(model.sample_rate), autoplay=False)