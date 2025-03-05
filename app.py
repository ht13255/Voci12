import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment, effects
import tempfile
import os

# AI 기반 최적화 파라미터 조정 함수
def ai_optimize_parameters(audio_seg):
    try:
        dBFS = audio_seg.dBFS  # 평균 음량 (dBFS)
        if dBFS < -30:
            boost_factor = 1.8
            lowpass_cutoff = 3500
        elif dBFS < -20:
            boost_factor = 1.5
            lowpass_cutoff = 4000
        else:
            boost_factor = 1.2
            lowpass_cutoff = 4500
    except Exception:
        boost_factor = 1.2
        lowpass_cutoff = 4000
    return {"boost_factor": boost_factor, "lowpass_cutoff": lowpass_cutoff}

# NumPy 배열을 AudioSegment (16-bit PCM)로 변환
def numpy_to_audiosegment(audio_array, sr):
    audio_array = np.int16(audio_array * 32767)
    return AudioSegment(
        audio_array.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

# 녹음 정제 (일반 녹음)
def process_recorded_audio(audio_array: np.ndarray, sr: int) -> AudioSegment:
    reduced_audio = nr.reduce_noise(y=audio_array, sr=sr, prop_decrease=0.5, stationary=True)
    audio_seg = numpy_to_audiosegment(reduced_audio, sr)
    params = ai_optimize_parameters(audio_seg)
    boost_factor = params["boost_factor"]
    lowpass_cutoff = params["lowpass_cutoff"]
    audio_seg = effects.normalize(audio_seg)
    audio_seg = effects.low_pass_filter(audio_seg, cutoff=lowpass_cutoff)
    audio_seg = effects.compress_dynamic_range(audio_seg)
    audio_seg = apply_bass_boost(audio_seg, boost_factor=boost_factor, cutoff=150)
    return effects.normalize(audio_seg)

# 현장 녹음 정제 (페스티벌 등)
def process_festival_audio(audio_array: np.ndarray, sr: int) -> AudioSegment:
    reduced_audio = nr.reduce_noise(y=audio_array, sr=sr, prop_decrease=0.6, stationary=True)
    audio_seg = numpy_to_audiosegment(reduced_audio, sr)
    params = ai_optimize_parameters(audio_seg)
    boost_factor = params["boost_factor"]
    lowpass_cutoff = params["lowpass_cutoff"]
    audio_seg = effects.normalize(audio_seg)
    audio_seg = effects.low_pass_filter(audio_seg, cutoff=lowpass_cutoff)
    audio_seg = effects.compress_dynamic_range(audio_seg)
    audio_seg = apply_bass_boost(audio_seg, boost_factor=boost_factor, cutoff=150)
    return effects.normalize(audio_seg)

# AI 자동 믹싱 (보컬 + MR 합성)
def mix_audio(vocal_array: np.ndarray, mr_array: np.ndarray, sr: int) -> AudioSegment:
    vocal = numpy_to_audiosegment(vocal_array, sr)
    mr = numpy_to_audiosegment(mr_array, sr)
    vocal = effects.normalize(vocal)
    vocal = effects.compress_dynamic_range(vocal)
    vocal = apply_bass_boost(vocal, boost_factor=1.1, cutoff=150)
    mixed = mr.overlay(vocal, position=0)
    return effects.normalize(mixed)

# AI 기반 베이스 부스트 (간단한 예제)
def apply_bass_boost(audio, boost_factor=1.2, cutoff=150):
    # boost_factor * 5 dB 만큼 증폭 후 low pass filter 적용
    return effects.low_pass_filter(audio + boost_factor * 5, cutoff=cutoff)

def main():
    st.title("🎶 AI 음원 정제 & 자동 믹싱 프로그램")
    st.write("사이드바에서 원하는 작업을 선택하세요.")

    # 사이드바를 활용한 작업 선택
    option = st.sidebar.radio(
        "작업 선택",
        ("🎧 녹음 정제", "🎤 현장 녹음 정제", "🎼 AI 믹싱")
    )

    # 파일 경로 변수 초기화
    file_path = None
    mr_path = None
    processed_audio = None

    # 모든 모드에서 공통으로 오디오 파일 업로드
    uploaded_file = st.file_uploader("🎤 오디오 파일을 업로드하세요", type=["wav", "mp3"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.read())
            file_path = temp_file.name

        try:
            audio_array, sr = librosa.load(file_path, sr=44100)
        except Exception as e:
            st.error(f"오디오 로드 오류: {e}")
            return

        if option == "🎧 녹음 정제":
            st.write("🔹 AI가 녹음된 음성을 깨끗하게 정제합니다.")
            try:
                processed_audio = process_recorded_audio(audio_array, sr)
            except Exception as e:
                st.error(f"녹음 정제 오류: {e}")

        elif option == "🎤 현장 녹음 정제":
            st.write("🔹 AI가 페스티벌 녹음의 노이즈를 정밀하게 제거합니다.")
            try:
                processed_audio = process_festival_audio(audio_array, sr)
            except Exception as e:
                st.error(f"현장 녹음 정제 오류: {e}")

        elif option == "🎼 AI 믹싱":
            st.write("🔹 AI가 MR과 보컬을 자동으로 믹싱합니다.")
            mr_file = st.file_uploader("🎶 MR 파일을 업로드하세요", type=["wav", "mp3"])
            if mr_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_mr:
                    temp_mr.write(mr_file.read())
                    mr_path = temp_mr.name
                try:
                    mr_array, _ = librosa.load(mr_path, sr=sr)
                    processed_audio = mix_audio(audio_array, mr_array, sr)
                except Exception as e:
                    st.error(f"AI 믹싱 오류: {e}")
            else:
                st.warning("MR 파일을 업로드하세요.")

        if processed_audio:
            output_path = "processed_audio.wav"
            try:
                processed_audio.export(output_path, format="wav")
                st.audio(output_path, format="audio/wav")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 다운로드",
                        data=f.read(),
                        file_name="processed_audio.wav",
                        mime="audio/wav"
                    )
            except Exception as e:
                st.error(f"출력 파일 생성 오류: {e}")
            finally:
                if os.path.exists(output_path):
                    os.remove(output_path)

    # 임시 파일 정리
    if file_path is not None and os.path.exists(file_path):
        os.remove(file_path)
    if mr_path is not None and os.path.exists(mr_path):
        os.remove(mr_path)

if __name__ == '__main__':
    main()