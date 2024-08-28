import torch
import numpy as np

framelength = 320
win_shift = 80
EPSILON = 1e-10
win1 = np.ones(framelength)
win2 = np.ones(framelength)


# 非对称窗
for i in range(framelength):
    if i < 3.0 / 4.0 * framelength:
        win1[i] = np.sqrt(1.0 / 2.0 * (1.0 - np.cos(2.0 * np.pi * i / (3.0 / 2.0 * framelength))))
    else:
        win1[i] = np.sqrt(1.0 / 2.0 * (1.0 - np.cos(2.0 * np.pi * (i - framelength / 2.0) / (1.0 / 2.0 * framelength))))
#
for i in range(framelength):
    if i < 3.0 / 4.0 * framelength and i >= 1.0 / 2.0 * framelength:
        win2[i] = 1.0 / 2.0 * (1.0 - np.cos(2.0 * np.pi * (i - framelength / 2.0) / (1.0 / 2.0 * framelength))) / np.sqrt(1.0 / 2.0 * (1.0 - np.cos(2.0 * np.pi * i / (3.0 / 2.0 * framelength))))
    elif i >= 3.0 / 4.0 * framelength and i < framelength:
        win2[i] = np.sqrt(1.0 / 2.0 * (1.0 - np.cos(2.0 * np.pi * (i - framelength / 2.0) / (1.0 / 2.0 * framelength))))
    else:
        win2[i] = 0

win1 = torch.from_numpy(win1)
win2 = torch.from_numpy(win2)



def asymmetric_stft_batch(batch_sig_wav, win_size=framelength, win_shift=win_shift, fft_num=framelength):
    """
    :param batch_sig_wav: shape[B, t]
    :param win_size: 320
    :param win_shift: 80
    :param fft_num: 320
    :return: [B F T]
    """
    device = batch_sig_wav.device
    batch_size = batch_sig_wav.shape[0]

    win = win1.unsqueeze(-1).to(device)

    batch_sig_wav = torch.nn.functional.pad(batch_sig_wav, (win_size - win_shift, win_shift), mode='constant', value=0)


    # batch divide frame
    batch_sig_wav = batch_sig_wav[:, :batch_sig_wav.shape[1] // win_shift * win_shift]
    framed_signals = batch_sig_wav.unfold(1, win_size, win_shift).permute(0, 2, 1).to(device)

    # Apply window
    framed_signals = framed_signals * win
    framed_signals = framed_signals.float()
    # Compute STFT
    batch_sig_stft = torch.fft.rfft(framed_signals, fft_num, dim=1)

    # Compute magnitude and phase
    batch_sig_mag = torch.abs(batch_sig_stft)
    batch_sig_phase = torch.angle(batch_sig_stft)

    # Convert magnitude and phase to complex numbers
    batch_sig_stft_complex = batch_sig_mag * torch.exp(1j * batch_sig_phase)

    return batch_sig_stft_complex  # Shape [B, T, F]

def asymmetric_istft_batch(batch_est_stft, win_size=framelength, win_shift=win_shift):
    """
    :param batch_est_stft: shape[B F T]
    :param win_size: 320
    :param win_shift: 80
    :return: [B t]
    """
    B, F, T = batch_est_stft.shape
    device = batch_est_stft.device

    batch_est_stft = torch.view_as_real(batch_est_stft)  # (B F T 2)
    batch_est_stft = batch_est_stft.permute(0, 3, 2, 1)  # (B 2 T F)

    win = win2.unsqueeze(-1).to(device)

    batch_est_mag, batch_est_phase = (torch.norm(batch_est_stft, dim=1) + EPSILON), \
                                      torch.atan2(batch_est_stft[:, -1, ...] + EPSILON,
                                                  batch_est_stft[:, 0, ...] + EPSILON)
    batch_est_stft = torch.stack((batch_est_mag * torch.cos(batch_est_phase),
                                  batch_est_mag * torch.sin(batch_est_phase)), dim=1)  # (B,2,T,F)
    batch_est_stft = batch_est_stft[:, 0, :, :] + 1j * batch_est_stft[:, 1, :, :]  # (B,T,F)

    batch_size = batch_est_stft.shape[0]
    nfrms = batch_est_stft.shape[1]


    # Compute the IFFT for all frames
    batch_est_time = torch.fft.irfft(batch_est_stft, dim=-1)
    batch_est_time = batch_est_time[:,:,:win_size]
    # Apply the window
    batch_est_time *= win.permute(1,0).unsqueeze(0)

    # batch OLA
    batch_est = torch.zeros((batch_size, nfrms-1, win_shift), device=device, dtype=torch.float32)
    batch_est[:, :, :] = batch_est_time[:, :-1, win_size-win_shift:] + batch_est_time[:, 1:, win_size-2*win_shift:win_size-win_shift]
    batch_est = batch_est.permute(0, 1, 2).reshape(B, -1)

    return batch_est


if __name__ == "__main__":

    x = torch.randn((1, 3200)).cuda()
    print(x)
    x_stft = asymmetric_stft_batch(x)
    x_hat = asymmetric_istft_batch(x_stft)
    print(torch.max(torch.tensor(x_hat).cuda() - x))
    print(x_hat)


