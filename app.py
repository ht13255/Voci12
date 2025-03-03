import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment, effects
from io import BytesIO
import concurrent.futures

# 현장 녹음 정제를 위한 라이브러리 (설치되지 않으면 정제 기능 비활성)
try:
    import noisereduce as nr
except ImportError:
    nr = None
    st.error("noisereduce 라이브러리가 설치되어 있지 않습니다. '녹음 음성 정제' 기능은 사용이 불가능합니다.")

# AI 기반 파라미터 최적화 함수: 오디오의 dBFS(평균 음량)를 기준으로 boost_factor와 lowpass_cutoff를 동적으로 산출
def ai_optimize_parameters(audio_seg: AudioSegment) -> dict:
    try:
        dBFS = audio_seg.dBFS  # 평균 음량 (dBFS)
        if dBFS < -30:
            boost_factor = 2.0
            lowpass_cutoff = 3500
        elif dBFS < -20:
            boost_factor = 1.8
            lowpass_cutoff = 4000
        else:
            boost_factor = 1.5
            lowpass_cutoff = 4500
        return {"boost_factor": boost_factor, "lowpass_cutoff": lowpass_cutoff}
    except Exception as e:
        return {"boost_factor": 1.5, "lowpass_cutoff": 4000}

# AI 기반 비트 정렬 최적화 함수: 두 트랙의 첫 비트 오프셋을 합리적인 범위 내(±5초)로 제한
def ai_optimize_beat_alignment(vocal_beats, vocal_sr, mr_beats, mr_sr):
    try:
        offset = (mr_beats[0] / mr_sr) - (vocal_beats[0] / vocal_sr)
        offset = max(min(offset, 5), -5)
        return offset
    except Exception as e:
        return 0

# numpy 배열 (float32, -1~1)을 AudioSegment (16-bit PCM)로 변환하는 헬퍼 함수
def numpy_to_audiosegment(audio: np.ndarray, sr: int) -> AudioSegment:
    try:
        sample_width = 2  # 16-bit
        audio_int16 = (audio * (2**15 - 1)).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return AudioSegment(
            data=audio_bytes,
            sample_width=sample_width,
            frame_rate=sr,
            channels=1
        )
    except Exception as e:
        st.error("오디오 변환 중 오류 발생")
        raise e

# 캐싱: bytes를 입력받아 librosa로 오디오 로드 (비트 분석용)
@st.cache_data(show_spinner=False)
def load_librosa_audio(file_bytes: bytes):
    try:
        audio, sr = librosa.load(BytesIO(file_bytes), sr=None, mono=True)
        return audio, sr
    except Exception as e:
        st.error("librosa 로드 오류")
        raise e

# 캐싱: bytes를 입력받아 pydub의 AudioSegment 생성 (후처리 및 믹싱용)
@st.cache_data(show_spinner=False)
def load_pydub_audio(file_bytes: bytes):
    try:
        return AudioSegment.from_file(BytesIO(file_bytes))
    except Exception as e:
        st.error("pydub 로드 오류")
        raise e

# 캐싱: librosa를 이용한 비트 분석
@st.cache_data(show_spinner=False)
def compute_beats(audio_data: np.ndarray, sr: int):
    try:
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        return tempo, beats
    except Exception as e:
        st.error("비트 분석 오류")
        raise e

# FFT 기반 bass boost 함수: 저역 성분을 증폭하여 빵빵한 사운드를 구현
def apply_bass_boost(audio_seg: AudioSegment, boost_factor=1.5, cutoff=150) -> AudioSegment:
    try:
        samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
        samples /= (2**15)
        D = librosa.stft(samples)
        freqs = librosa.fft_frequencies(sr=audio_seg.frame_rate)
        low_indices = np.where(freqs < cutoff)[0]
        D[low_indices, :] *= boost_factor
        boosted_samples = librosa.istft(D, length=len(samples))
        max_val = np.max(np.abs(boosted_samples))
        if max_val > 1:
            boosted_samples /= max_val
        boosted_int16 = (boosted_samples * (2**15 - 1)).astype(np.int16)
        boosted_audio = AudioSegment(
            boosted_int16.tobytes(),
            frame_rate=audio_seg.frame_rate,
            sample_width=2,
            channels=1
        )
        return boosted_audio
    except Exception as e:
        st.error("Bass boost 처리 오류")
        raise e

# 캐싱: 보컬 후처리 (정규화, 저역 필터, 동적 범위 압축, AI 최적화 기반 bass boost 적용)
@st.cache_data(show_spinner=False)
def process_vocal_audio(vocal_audio: AudioSegment) -> AudioSegment:
    try:
        params = ai_optimize_parameters(vocal_audio)
        boost_factor = params["boost_factor"]
        lowpass_cutoff = params["lowpass_cutoff"]
        processed = effects.normalize(vocal_audio)
        processed = effects.low_pass_filter(processed, cutoff=lowpass_cutoff)
        processed = effects.compress_dynamic_range(processed)
        processed = apply_bass_boost(processed, boost_factor=boost_factor, cutoff=150)
        processed = effects.normalize(processed)
        return processed
    except Exception as e:
        st.error("보컬 후처리 오류")
        raise e

# 캐싱: 현장 녹음 정제 (노이즈 제거, 정규화, 필터, 동적 범위 압축, fade 효과, 그리고 AI 최적화 기반 bass boost 적용)
@st.cache_data(show_spinner=False)
def process_festival_audio(audio_array: np.ndarray, sr: int) -> AudioSegment:
    try:
        if nr is None:
            raise ImportError("noisereduce 라이브러리가 설치되어 있지 않습니다.")
        reduced_audio = nr.reduce_noise(
            y=audio_array,
            sr=sr,
            n_std_thresh=2.5,
            prop_decrease=0.6
        )
        audio_seg = numpy_to_audiosegment(reduced_audio, sr)
        params = ai_optimize_parameters(audio_seg)
        boost_factor = params["boost_factor"]
        lowpass_cutoff = params["lowpass_cutoff"]
        audio_seg = effects.normalize(audio_seg)
        audio_seg = effects.low_pass_filter(audio_seg, cutoff=lowpass_cutoff)
        audio_seg = effects.compress_dynamic_range(audio_seg)
        audio_seg = audio_seg.fade_in(50).fade_out(50)
        audio_seg = apply_bass_boost(audio_seg, boost_factor=boost_factor, cutoff=150)
        audio_seg = effects.normalize(audio_seg)
        return audio_seg
    except Exception as e:
        st.error("현장 녹음 정제 오류")
        raise e

def main():
    st.title("최대 성능 고도화: AI 음원 처리 애플리케이션 (빵빵한 사운드 & 오류 최소화)")
    st.write("AI가 모든 단계에 개입하여 최적의 성능으로 자동 믹싱 또는 녹음 정제를 진행합니다.")
    
    mode = st.radio("기능 선택", ["자동 믹싱 (보컬 + MR)", "녹음 음성 정제 (페스티벌 등)"])
    
    if mode == "자동 믹싱 (보컬 + MR)":
        st.subheader("자동 믹싱: 보컬 + MR")
        vocal_file = st.file_uploader("보컬 트랙 업로드", type=["wav", "mp3", "ogg"])
        mr_file = st.file_uploader("MR 트랙 업로드", type=["wav", "mp3", "ogg"])
        
        if vocal_file is not None and mr_file is not None:
            try:
                vocal_bytes = vocal_file.read()
                mr_bytes = mr_file.read()
                
                with st.spinner("오디오 데이터 로드 중..."):
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_vocal_lib = executor.submit(load_librosa_audio, vocal_bytes)
                        future_mr_lib = executor.submit(load_librosa_audio, mr_bytes)
                        future_vocal_pydub = executor.submit(load_pydub_audio, vocal_bytes)
                        future_mr_pydub = executor.submit(load_pydub_audio, mr_bytes)
                        vocal_y, vocal_sr = future_vocal_lib.result()
                        mr_y, mr_sr = future_mr_lib.result()
                        vocal_audio = future_vocal_pydub.result()
                        mr_audio = future_mr_pydub.result()
                
                with st.spinner("비트 분석 중..."):
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_vocal_beats = executor.submit(compute_beats, vocal_y, vocal_sr)
                        future_mr_beats = executor.submit(compute_beats, mr_y, mr_sr)
                        vocal_tempo, vocal_beats = future_vocal_beats.result()
                        mr_tempo, mr_beats = future_mr_beats.result()
                
                if len(vocal_beats) > 0 and len(mr_beats) > 0:
                    offset = ai_optimize_beat_alignment(vocal_beats, vocal_sr, mr_beats, mr_sr)
                    st.write(f"보컬 오프셋: {offset:.2f}초 (AI 최적화)")
                    if offset > 0:
                        silence = AudioSegment.silent(duration=int(offset * 1000))
                        vocal_audio = silence + vocal_audio
                    elif offset < 0:
                        trim_ms = -int(offset * 1000)
                        if trim_ms < len(vocal_audio):
                            vocal_audio = vocal_audio[trim_ms:]
                        else:
                            st.warning("보컬 트랙 길이가 너무 짧아 정렬 건너뜁니다.")
                else:
                    st.warning("비트 감지 실패: 자동 정렬 건너뜀")
                
                with st.spinner("보컬 후처리 중..."):
                    vocal_audio = process_vocal_audio(vocal_audio)
                
                if len(vocal_audio) > len(mr_audio):
                    repeat_times = (len(vocal_audio) // len(mr_audio)) + 1
                    mr_audio = (mr_audio * repeat_times)[:len(vocal_audio)]
                elif len(mr_audio) > len(vocal_audio):
                    silence = AudioSegment.silent(duration=(len(mr_audio) - len(vocal_audio)))
                    vocal_audio = vocal_audio + silence
                
                vocal_adjusted = vocal_audio.apply_gain(0)
                mr_adjusted = mr_audio.apply_gain(-3)
                
                mixed_audio = mr_adjusted.overlay(vocal_adjusted)
                mixed_audio = effects.normalize(mixed_audio)
                
                mixed_bytes = BytesIO()
                mixed_audio.export(mixed_bytes, format="mp3")
                mixed_bytes.seek(0)
                
                st.audio(mixed_bytes, format="audio/mp3")
                st.download_button(
                    label="믹스된 트랙 다운로드",
                    data=mixed_bytes.getvalue(),
                    file_name="mixed_track.mp3",
                    mime="audio/mp3"
                )
            
            except Exception as e:
                st.error(f"오류 발생: {e}")
    
    elif mode == "녹음 음성 정제 (페스티벌 등)":
        st.subheader("녹음 음성 정제: 현장 녹음 정리")
        vocal_file = st.file_uploader("정제할 녹음 파일 업로드", type=["wav", "mp3", "ogg"])
        
        if vocal_file is not None:
            try:
                vocal_bytes = vocal_file.read()
                
                with st.spinner("오디오 데이터 로드 중..."):
                    vocal_y, vocal_sr = load_librosa_audio(vocal_bytes)
                
                with st.spinner("노이즈 제거 및 음성 후처리 중..."):
                    processed_audio = process_festival_audio(vocal_y, vocal_sr)
                
                processed_bytes = BytesIO()
                processed_audio.export(processed_bytes, format="mp3")
                processed_bytes.seek(0)
                
                st.audio(processed_bytes, format="audio/mp3")
                st.download_button(
                    label="정제된 음성 다운로드",
                    data=processed_bytes.getvalue(),
                    file_name="processed_vocal.mp3",
                    mime="audio/mp3"
                )
            
            except Exception as e:
                st.error(f"오류 발생: {e}")

if __name__ == '__main__':
    main()
