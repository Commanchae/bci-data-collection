import os
import time

import numpy as np
import datetime as dt

from pylsl import resolve_byprop, StreamInfo, StreamInlet
from pygame import mixer
from typing import List, Callable
from collections.abc import Iterable

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
mixer.init()

class DataRecorder():
    def __init__(self, sampling_frequency: int=256, additional_variables: List[str]=[]) -> None:
        self.sampling_frequency = sampling_frequency
        self.samples = {
            'recordings': []
        }
        for variable in additional_variables:
            self.samples[variable] = []

    # Can be overridden.
    def signal_start_recording(self):
        pass

    # Can be overridden.
    def signal_end_recording(self):
        pass

    # Can be overridden.
    def present_stimuli(self):
        pass

    # Can be overridden.
    def end_stimuli(self):
        pass

    # Should be overridden.
    def record_one_iteration(self, stream: StreamInfo, duration: float, rest_duration: float, sampling_frequency: float):
        '''
        Inputs:
            1. stream: StreamInfo object corresponding to the EEG stream.
            2. duration: Duration of the recording in seconds.
            3. rest_duration: The time before recording starts (to give users a break)
        
        Output:
            1. sample: An np.ndarray of shape [C, T]
                where C corresponds to the number of EEG channels in the stream.
                and T refers to the timesteps.
        '''
        time.sleep(rest_duration)

        # Presents stimuli to the user.
        self.present_stimuli()

        # Records user's response to stimuli.
        sample = self._record_eeg_sample(stream, duration)

        # Stops presenting the stimuli.
        self.end_stimuli()

        # Pad / trims the sample to have the desired timesteps (sampling_frequency * duration). Ex: 256 for 2 seconds would result in 512 datapoints.
        sample = self._padtrim_sample(sample, sampling_frequency=sampling_frequency, duration=duration)
        return sample



    def record_loop(self, stream: StreamInfo=None, iterations: int=10, duration: float=2.0, rest_duration: float=2.0, additional_variables: dict=dict()) -> None:
        '''
        Inputs:
            1. stream: StreamInfo object corresponding to the EEG stream.
            2. iterations: Number of iterations to record.
            3. duration: Duration of each recording in seconds.
            4. rest_duration: Duration of rest time between recordings in seconds.

        Outputs:
            None. Appends EEG samples to self.samples['recordings'].
        '''

        # If no stream is provided, initiate a new stream.
        if stream is None:
            stream = self._initiate_stream()
            if stream is None:
                print("No EEG stream was found.")
                return
        
        # >> A stream has been found and connected.
        # First, let's make sure the additional variables are valid.
        list_variables = []
        object_variables = []

        if len(additional_variables) > 0:
            for key in additional_variables:
                if isinstance(additional_variables[key], Iterable):
                    assert len(additional_variables[key]) == iterations, f'If a list is provided for \'{key}\', then the length of the list must be equal to the number of iterations.'
                    list_variables.append(key)
                else:
                    object_variables.append(key)
        
        # >> Additional variables are valid.
        # Now, we can start recording.
        self.signal_start_recording()

        for i in range(iterations):
            sample = self.record_one_iteration(stream, duration=duration, rest_duration=rest_duration, sampling_frequency=self.sampling_frequency)
            self.samples['recordings'].append(sample)
        self.signal_end_recording()    

        # Now, we append the additional variables to our lists.
        for key in list_variables:
            self.samples[key].extend(additional_variables[key])
        
        for key in object_variables:
            self.samples[key].extend([additional_variables[key] * iterations])
    

    def _initiate_stream(self, timeout: int=4) -> StreamInfo | None:
        '''
        Initiates a stream with (timeout) timeout. If a stream has not been found within
        the specified timeout seconds, then it is assumed no stream is present.
        '''

        streams = resolve_byprop('type', 'EEG', timeout=timeout)
        if not streams:
            return None
        return streams
    
    def _record_eeg_sample(self, stream: StreamInfo, duration: float) -> np.ndarray:
        '''
        Inputs: stream channel and duration of recording (in seconds).
            Records a single EEG recording from (stream) stream channel for (duration) seconds.

        Outputs: A numpy array of shape [C, duration*sampling_frequency]. Where C refers to the recording channels in the connection (5 for Muse2).
        '''
        # Gets stream inlet.
        inlet = StreamInlet(stream[0])

        # Initialize required variables to calculate time elapsed and compile samples.
        start_time = dt.datetime.now()
        samples = []

        # Collect data for the given duration.
        while dt.datetime.now() - start_time < dt.timedelta(seconds=duration):
            sample, _ = inlet.pull_sample()
            samples.append(sample) 

        samples = np.array(samples) # Shape: [duration*sampling_frequency, C] (C=5 channels from Muse 2). Example shape: [512, 5]
        samples = samples.T # Shape: [C, duration*sampling_frequency]. Example shape: [5, 512]
        samples = np.ascontiguousarray(samples) # Converts array to contiguous array for faster processing.
        return samples

    def _padtrim_sample(self, samples: np.ndarray, sampling_frequency: int, duration: float) -> np.ndarray:
        '''
        Inputs: 
            samples (np.ndarray) of shape [C, ANY LENGTH]
                If ANY LENGTH is less than sampling_frequency * duration, then the sample will be padded with 0s at the end.
                Else, the sample will be trimmed to be exactly sampling_frequency * duration in shape (dim=1).
            sampling_frequency of the EEG headset. 
            duration of the recording (in seconds)

        Output:
            A padded or trimmed version of samples (padding of 0 at the end if the samples)
        '''

        # Get the number of channels and timesteps.
        C, T = samples.shape

        # Calculate the desired length.
        target_length = int(sampling_frequency * duration)

        # If the sample is shorter than the desired length, pad it.
        if T < target_length:
            pad_width = ((0, 0), (0, target_length - T))
            samples = np.pad(samples, pad_width, 'constant')
        else:
            # Else, if the sample is longer than the desired length, trim it.
            samples = samples[:, :target_length]
        
        # Return the padded/trimmed sample.
        return samples


# class DataRecorder():
#     def __init__(self, sampling_frequency: int=256, sound_required: bool=False) -> None:
#         self.sampling_frequency = sampling_frequency

#         if sound_required:
#             self.sound_dictionary = {
#                 "enter": mixer.Sound("sounds/enter.mp3"),
#                 "bell": mixer.Sound("sounds/bell.mp3")
#             }
#         else:
#             self.sound_dictionary = {}
        
#         self.samples = []
#         self.actions = []
#         self.users = []

#     def record_loop(self, duration: float=2.0, iterations: int=10, rest_duration: int=2, stream: StreamInfo=None,action: str = 'Nothing') -> bool:
#         '''
#         Records a (duration) seconds EEG sample for (iterations) iterations.

#         Outputs: Boolean that states whether recording was successful. Appends to self.samples 10 (iterations) np.ndarrays of shape [5, duration*self.sampling_frequency]. 
#                     5 corresponds to the number of electrode channels present in the Muse 2. (For usage, we have to remove the last channel, as it's irrelevant.)
        
#         Call self.save_samples(filepath) to save samples and actions into one .npz file.
#         '''
#         # Test whether a stream can be found.
#         if stream is None:
#             stream = self._initiate_stream()
#             if stream is None:
#                 print("No EEG stream was found.")
#                 return False
        
#         # >> A stream has been found and connected.
#         # This sound signifies the start of the trial.
#         self._play_sound("bell")

#         for i in range(iterations):
#             time.sleep(rest_duration) # This 2 seconds duration is used for rest time, where no brain activity is recorded. This allows for participants to perform other actions, such as blinking and heavy breathing.
#             sample = self._record_eeg_sample(stream, duration)
#             sample = self._padtrim_sample(sample, self.sampling_frequency, duration)

#             self.samples.append(sample) # Directly modifies instance list.
#             self.actions.append(action) # Directly modifies instance list.

#             self._play_sound("enter") # This indicates the end of a single recording.
#         self._play_sound("bell") # This indicates the end of the trial.
#         return True

#     def _initiate_stream(self, timeout: int=4) -> StreamInfo | None:
#         '''
#         Initiates a stream with (timeout) timeout. If a stream has not been found within
#         the specified timeout seconds, then it is assumed no stream is present.
#         '''

#         streams = resolve_byprop('type', 'EEG', timeout=timeout)
#         if not streams:
#             return None
#         return streams

#     def _record_eeg_sample(self, stream: StreamInfo, duration: float) -> np.ndarray:
#         '''
#         Inputs: stream channel and duration of recording (in seconds).
#             Records a single EEG recording from (stream) stream channel for (duration) seconds.

#         Outputs: A numpy array of shape [C, duration*sampling_frequency]. Where C refers to the recording channels in the connection (5 for Muse2).
#         '''
#         # Gets stream inlet.
#         inlet = StreamInlet(stream[0])

#         # Initialize required variables to calculate time elapsed and compile samples.
#         start_time = dt.datetime.now()
#         samples = []

#         # Collect data for the given duration.
#         while dt.datetime.now() - start_time < dt.timedelta(seconds=duration):
#             sample, _ = inlet.pull_sample()
#             samples.append(sample) 

#         samples = np.array(samples) # Shape: [duration*sampling_frequency, C] (C=5 channels from Muse 2). Example shape: [512, 5]
#         samples = samples.T # Shape: [C, duration*sampling_frequency]. Example shape: [5, 512]
#         samples = np.ascontiguousarray(samples) # Converts array to contiguous array for faster processing.
#         return samples

#     def _padtrim_sample(self, samples: np.ndarray, sampling_frequency: int, duration: float) -> np.ndarray:
#         '''
#         Inputs: 
#             samples (np.ndarray) of shape [C, ANY LENGTH]
#                 If ANY LENGTH is less than sampling_frequency * duration, then the sample will be padded with 0s at the end.
#                 Else, the sample will be trimmed to be exactly sampling_frequency * duration in shape (dim=1).
#             sampling_frequency of the EEG headset. 
#             duration of the recording (in seconds)

#         Output:
#             A padded or trimmed version of samples (padding of 0 at the end if the samples)
#         '''

#         # Get the number of channels and timesteps.
#         C, T = samples.shape

#         # Calculate the desired length.
#         target_length = int(sampling_frequency * duration)

#         # If the sample is shorter than the desired length, pad it.
#         if T < target_length:
#             pad_width = ((0, 0), (0, target_length - T))
#             samples = np.pad(samples, pad_width, 'constant')
#         else:
#             # Else, if the sample is longer than the desired length, trim it.
#             samples = samples[:, :target_length]
        
#         # Return the padded/trimmed sample.
#         return samples