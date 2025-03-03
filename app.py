import streamlit as st
import numpy as np
import librosa
import librosa.display
import noisereduce as nr
from pydub import AudioSegment, effects
import tempfile
import os

# 🎵 AI 기반 최적화 파라미터 조정
def ai_optimize_parameters(audio_seg):
    # AI가 자동으로 파라미터 조정 (기본값 설정)
    return {
        "boost_factor": 1.2,  # 베이스 부스트 강도
        "lowpass_cutoff": 300  # 저역 필터 컷오프 주파수
    }

# 🎧 NumPy 배열을 AudioSegment로 변환
def numpy_to_audiosegment(audio_array, sr):
    audio_array = np.int16(audio_array * 32767)  # 16-bit PCM으로 변환
    return AudioSegment(audio_array.tobytes(), frame_rate=sr, sample_width=2, channels=1)

# 🎤 노이즈 제거 + 음성 정제 (일반 녹음)
def process_recorded_audio(audio_array: np.ndarray, sr: int) -> AudioSegment:
    reduced_audio = nr.reduce_noise(y=audio_array, sr=sr, prop_decrease=0.5, stationary=True)
    audio_seg = numpy_to_audiosegment(reduced_audio, sr)

    # 🔹 AI 자동 최적화
    params = ai_optimize_parameters(audio_seg)
    boost_factor = params["boost_factor"]
    lowpass_cutoff = params["lowpass_cutoff"]

    audio_seg = effects.normalize(audio_seg)
    audio_seg = effects.low_pass_filter(audio_seg, cutoff=lowpass_cutoff)
    audio_seg = effects.compress_dynamic_range(audio_seg)
    audio_seg = apply_bass_boost(audio_seg, boost_factor=boost_factor, cutoff=150)
    
    return effects.normalize(audio_seg)

# 🎶 페스티벌 등 현장 녹음 정제
def process_festival_audio(audio_array: np.ndarray, sr: int) -> AudioSegment:
    reduced_audio = nr.reduce_noise(y=audio_array, sr=sr, prop_decrease=0.6, stationary=True)
    audio_seg = numpy_to_audiosegment(reduced_audio, sr)

    # 🔹 AI 자동 최적화
    params = ai_optimize_parameters(audio_seg)
    boost_factor = params["boost_factor"]
    lowpass_cutoff = params["lowpass_cutoff"]

    audio_seg = effects.normalize(audio_seg)
    audio_seg = effects.low_pass_filter(audio_seg, cutoff=lowpass_cutoff)
    audio_seg = effects.compress_dynamic_range(audio_seg)
    audio_seg = apply_bass_boost(audio_seg, boost_factor=boost_factor, cutoff=150)

    return effects.normalize(audio_seg)

# 🎵 AI 자동 믹싱 (녹음 + MR 합성)
def mix_audio(vocal_array: np.ndarray, mr_array: np.ndarray, sr: int) -> AudioSegment:
    vocal = numpy_to_audiosegment(vocal_array, sr)
    mr = numpy_to_audiosegment(mr_array, sr)

    # 🔹 AI 자동 보정 (EQ + Compressor + Normalization)
    vocal = effects.normalize(vocal)
    vocal = effects.compress_dynamic_range(vocal)
    vocal = apply_bass_boost(vocal, boost_factor=1.1, cutoff=150)

    # 🔹 믹싱 후 음량 조정
    mixed = mr.overlay(vocal, position=0)
    return effects.normalize(mixed)

# 🔊 AI 기반 베이스 부스트
def apply_bass_boost(audio, boost_factor=1.2, cutoff=150):
    return effects.low_pass_filter(audio + boost_factor * 5, cutoff=cutoff)

# 🔥 Streamlit UI
st.title("🎶 AI 기반 음성 정제 & 자동 믹싱 프로그램")

uploaded_file = st.file_uploader("🎤 오디오 파일을 업로드하세요", type=["wav", "mp3"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    # 🎵 오디오 로드
    audio_array, sr = librosa.load(file_path, sr=44100)

    # 🔄 기능 선택
    option = st.radio("어떤 작업을 수행하시겠습니까?", ["🎧 녹음 정제", "🎤 현장 녹음 정제", "🎼 AI 믹싱"])

    if option == "🎧 녹음 정제":
        st.write("🔹 AI가 녹음된 음성을 깨끗하게 정제합니다.")
        processed_audio = process_recorded_audio(audio_array, sr)

    elif option == "🎤 현장 녹음 정제":
        st.write("🔹 AI가 페스티벌 녹음의 노이즈를 정밀하게 제거합니다.")
        processed_audio = process_festival_audio(audio_array, sr)

    elif option == "🎼 AI 믹싱":
        st.write("🔹 AI가 MR과 보컬을 자동으로 믹싱합니다.")
        mr_file = st.file_uploader("🎶 MR 파일을 업로드하세요", type=["wav", "mp3"])

        if mr_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_mr:
                temp_mr.write(mr_file.read())
                mr_path = temp_mr.name

            mr_array, _ = librosa.load(mr_path, sr=sr)
            processed_audio = mix_audio(audio_array, mr_array, sr)
        else:
            st.warning("MR 파일을 업로드하세요.")
            processed_audio = None

    if processed_audio:
        # 파일 저장 및 다운로드 제공
        output_path = "processed_audio.wav"
        processed_audio.export(output_path, format="wav")

        st.audio(output_path, format="audio/wav")
        st.download_button(label="📥 다운로드", data=open(output_path, "rb"), file_name="processed_audio.wav", mime="audio/wav")

# 🗑️ 임시 파일 정리
if os.path.exists(file_path):
    os.remove(file_path)
if os.path.exists("processed_audio.wav"):
    os.remove("processed_audio.wav")