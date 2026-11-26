// c:\Users\teeja\Downloads\FPS ChessPS\main.cpp
// Language: C++20 | File: main.cpp | Runtime: Win32 Native + Direct3D 11 + Dear ImGui | Bitness: x64
// Architecture: Read-only external process instrumentation (PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION),
//               cached reordered-IL2CPP FieldInfo reflection + direct RVA static tables (<0.22 ms/frame entity pass),
//               native Unity TransformHierarchy SIMD world-space bone solver, 1000 Hz precision SendInput controller,
//               and 1:1 Cloudy 1.3 spring-animated Direct3D 11 overlay shell rebranded for RELINQUISH.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>
#include <dwmapi.h>
#include <mmsystem.h>
#include <wincodec.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"
#include "imgui/imgui_impl_win32.h"
#include "imgui/imgui_impl_dx11.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "advapi32.lib")

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace fs = std::filesystem;

// ============================================================================
// 1. VECTOR / MATRIX / QUATERNION MATH
// ============================================================================
struct Vec2 {
    float x = 0.0f, y = 0.0f;
    constexpr Vec2() = default;
    constexpr Vec2(float _x, float _y) : x(_x), y(_y) {}
    Vec2 operator+(const Vec2& o) const { return { x + o.x, y + o.y }; }
    Vec2 operator-(const Vec2& o) const { return { x - o.x, y - o.y }; }
    Vec2 operator*(float s) const { return { x * s, y * s }; }
    float Length() const { return std::sqrt(x * x + y * y); }
};

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    constexpr Vec3() = default;
    constexpr Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    Vec3 operator+(const Vec3& o) const { return { x + o.x, y + o.y, z + o.z }; }
    Vec3 operator-(const Vec3& o) const { return { x - o.x, y - o.y, z - o.z }; }
    Vec3 operator*(float s) const { return { x * s, y * s, z * s }; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    float Length() const { return std::sqrt(x * x + y * y + z * z); }
    float Length2D() const { return std::sqrt(x * x + z * z); }
    float Dist(const Vec3& o) const { return (*this - o).Length(); }
    float Dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 Cross(const Vec3& o) const {
        return { y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x };
    }
    Vec3 Normalized() const {
        float l = Length();
        return (l > 1e-6f) ? Vec3(x / l, y / l, z / l) : Vec3(0.0f, 0.0f, 0.0f);
    }
    bool IsFinite() const {
        return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) &&
               std::fabs(x) < 50000.0f && std::fabs(y) < 50000.0f && std::fabs(z) < 50000.0f;
    }
};

struct Quat {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 1.0f;
};

struct Mat4 {
    // Column-major 4x4 float matrix matching Unity native Matrix4x4f
    float m[16]{};

    static Mat4 Mul(const Mat4& a, const Mat4& b) {
        Mat4 r{};
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                r.m[col * 4 + row] =
                    a.m[0 * 4 + row] * b.m[col * 4 + 0] +
                    a.m[1 * 4 + row] * b.m[col * 4 + 1] +
                    a.m[2 * 4 + row] * b.m[col * 4 + 2] +
                    a.m[3 * 4 + row] * b.m[col * 4 + 3];
            }
        }
        return r;
    }

    bool WorldToScreen(const Vec3& world, float screenW, float screenH, Vec2& outScreen) const {
        float clipX = m[0] * world.x + m[4] * world.y + m[8]  * world.z + m[12];
        float clipY = m[1] * world.x + m[5] * world.y + m[9]  * world.z + m[13];
        float clipW = m[3] * world.x + m[7] * world.y + m[11] * world.z + m[15];
        if (clipW < 0.05f || !std::isfinite(clipW)) return false;

        float invW = 1.0f / clipW;
        float ndcX = clipX * invW;
        float ndcY = clipY * invW;
        if (!std::isfinite(ndcX) || !std::isfinite(ndcY)) return false;

        outScreen.x = (screenW * 0.5f) * (1.0f + ndcX);
        outScreen.y = (screenH * 0.5f) * (1.0f - ndcY);
        return (outScreen.x > -screenW && outScreen.x < screenW * 2.0f &&
                outScreen.y > -screenH && outScreen.y < screenH * 2.0f);
    }

    // Offscreen directional projection preserving behind-camera sign
    void WorldToRadarClip(const Vec3& world, float& clipX, float& clipY, float& clipW) const {
        clipX = m[0] * world.x + m[4] * world.y + m[8]  * world.z + m[12];
        clipY = m[1] * world.x + m[5] * world.y + m[9]  * world.z + m[13];
        clipW = m[3] * world.x + m[7] * world.y + m[11] * world.z + m[15];
    }
};

// ============================================================================
// 2. FONT & LUCIDE ICON BANK (1:1 CLOUDY 1.3 GLYPH MAPPING)
// ============================================================================
#define ICON_MIN_LC 0xe000
#define ICON_MAX_LC 0xe700

#define ICON_FA_CROSSHAIRS     "\xee\x82\xab" // U+E0AB crosshair
#define ICON_FA_EYE            "\xee\x83\x80" // U+E0C0 eye
#define ICON_FA_SHIELD         "\xee\x85\x98" // U+E158 shield
#define ICON_FA_BOLT           "\xee\x86\xb4" // U+E1B4 zap
#define ICON_FA_SLIDERS        "\xee\x85\xa2" // U+E162 sliders
#define ICON_FA_FOLDER         "\xee\x83\x91" // U+E0D1 folder
#define ICON_FA_GEAR           "\xee\x85\x94" // U+E154 settings
#define ICON_FA_MAGNIFYING     "\xee\x85\x91" // U+E151 search
#define ICON_FA_CHECK          "\xee\x81\xb0" // U+E070 check
#define ICON_FA_XMARK          "\xee\x86\xb2" // U+E1B2 x
#define ICON_FA_CHEVRON_RIGHT  "\xee\x81\xb5" // U+E075 chevron-right
#define ICON_FA_CHEVRON_DOWN   "\xee\x81\xb3" // U+E073 chevron-down
#define ICON_FA_ROTATE         "\xee\x85\x84" // U+E144 refresh-cw
#define ICON_FA_FLOPPY         "\xee\x85\x8d" // U+E14D save
#define ICON_FA_TRASH          "\xee\x86\x8d" // U+E18D trash-2
#define ICON_FA_PLUS           "\xee\x85\x81" // U+E141 plus
#define ICON_FA_LINK           "\xee\x84\x88" // U+E108 link
#define ICON_FA_VOLUME         "\xee\x86\xa9" // U+E1A9 volume-2
#define ICON_FA_PALETTE        "\xee\x87\x9d" // U+E1DD palette
#define ICON_FA_SPARKLES       "\xee\x90\x96" // U+E416 sparkles
#define ICON_FA_RADAR          "\xee\x84\xb9" // U+E139 radio/radar
#define ICON_FA_SKULL          "\xee\x93\xa2" // U+E4E2 skull
#define ICON_FA_FLAME          "\xee\x83\x8e" // U+E0CE flame

struct FontBank {
    ImFont* reg14    = nullptr;
    ImFont* reg16    = nullptr;
    ImFont* med15    = nullptr;
    ImFont* med17    = nullptr;
    ImFont* bold16   = nullptr;
    ImFont* bold20   = nullptr;
    ImFont* bold28   = nullptr;
    ImFont* esp13    = nullptr;
    ImFont* espBold13 = nullptr;

    void Load(ImGuiIO& io, const fs::path& assetDir) {
        static const ImWchar iconRanges[] = { ICON_MIN_LC, ICON_MAX_LC, 0 };
        fs::path pReg  = assetDir / "Satoshi-Regular.ttf";
        fs::path pMed  = assetDir / "Satoshi-Medium.ttf";
        fs::path pBold = assetDir / "Satoshi-Bold.ttf";
        fs::path pIcon = assetDir / "lucide.ttf";

        bool hasFonts = fs::exists(pReg) && fs::exists(pMed) && fs::exists(pBold);
        bool hasIcons = fs::exists(pIcon);

        auto loadWithIcons = [&](const fs::path& mainFont, float sizePx, float iconScale = 0.90f) -> ImFont* {
            ImFontConfig cfg{};
            cfg.OversampleH = 3;
            cfg.OversampleV = 2;
            ImFont* base = nullptr;
            if (hasFonts && fs::exists(mainFont)) {
                base = io.Fonts->AddFontFromFileTTF(mainFont.string().c_str(), sizePx, &cfg);
            } else {
                cfg.SizePixels = sizePx;
                base = io.Fonts->AddFontDefault(&cfg);
            }
            if (hasIcons) {
                ImFontConfig icfg{};
                icfg.MergeMode = true;
                icfg.PixelSnapH = true;
                icfg.GlyphMinAdvanceX = sizePx * iconScale;
                icfg.GlyphOffset = ImVec2(0.0f, 1.5f);
                io.Fonts->AddFontFromFileTTF(pIcon.string().c_str(), sizePx * iconScale, &icfg, iconRanges);
            }
            return base;
        };

        med15     = loadWithIcons(pMed,  15.5f, 0.92f);
        reg14     = loadWithIcons(pReg,  14.0f, 0.90f);
        reg16     = loadWithIcons(pReg,  16.0f, 0.90f);
        med17     = loadWithIcons(pMed,  17.0f, 0.92f);
        bold16    = loadWithIcons(pBold, 16.0f, 0.92f);
        bold20    = loadWithIcons(pBold, 20.5f, 0.92f);
        bold28    = loadWithIcons(pBold, 28.0f, 0.90f);
        esp13     = loadWithIcons(pMed,  13.5f, 0.88f);
        espBold13 = loadWithIcons(pBold, 13.5f, 0.88f);
        io.FontDefault = med15;
    }
};

static FontBank g_Fonts;

// ============================================================================
// 3. SPRING-PHYSICS ANIMATION ENGINE & POLYPHONIC SOUND ENGINE (1:1 SHELL)
// ============================================================================
enum class SpringPreset : int {
    Soft   = 0,
    Snappy = 1,
    Bouncy = 2,
    Smooth = 3
};

struct SpringState {
    float value = 0.0f;
    float vel   = 0.0f;
    bool  init  = false;
};

class AnimEngine {
public:
    SpringPreset preset = SpringPreset::Snappy;
    float speedScale = 1.0f;

    float Step(ImGuiID id, float target, float dt, SpringPreset overridePreset = SpringPreset::Snappy, bool useOverride = false) {
        auto& st = states_[id];
        if (!st.init) {
            st.value = target;
            st.vel   = 0.0f;
            st.init  = true;
            return st.value;
        }
        SpringPreset p = useOverride ? overridePreset : preset;
        float omega = 18.0f;
        float zeta  = 0.82f;
        switch (p) {
            case SpringPreset::Soft:   omega = 12.5f; zeta = 0.92f; break;
            case SpringPreset::Snappy: omega = 22.0f; zeta = 0.80f; break;
            case SpringPreset::Bouncy: omega = 24.0f; zeta = 0.52f; break;
            case SpringPreset::Smooth: omega = 15.5f; zeta = 1.00f; break;
        }
        omega *= std::clamp(speedScale, 0.35f, 2.5f);
        float cdt = std::min(dt, 0.033f);
        float f = 1.0f + 2.0f * cdt * zeta * omega;
        float oo = omega * omega;
        float hoo = cdt * oo;
        float hhoo = cdt * hoo;
        float detInv = 1.0f / (f + hhoo);
        float detX = f * st.value + cdt * st.vel + hhoo * target;
        float detV = st.vel + hoo * (target - st.value);
        st.value = detX * detInv;
        st.vel   = detV * detInv;
        if (std::fabs(st.value - target) < 0.0005f && std::fabs(st.vel) < 0.002f) {
            st.value = target;
            st.vel   = 0.0f;
        }
        return st.value;
    }

    float StepKey(const char* key, float target, float dt) {
        return Step(ImHashStr(key), target, dt);
    }

    ImVec4 LerpColor(const char* key, const ImVec4& target, float dt) {
        ImGuiID base = ImHashStr(key);
        return ImVec4(
            Step(base + 1, target.x, dt),
            Step(base + 2, target.y, dt),
            Step(base + 3, target.z, dt),
            Step(base + 4, target.w, dt)
        );
    }

private:
    std::unordered_map<ImGuiID, SpringState> states_;
};

static AnimEngine g_Anim;

enum class SoundPack : int {
    Pop       = 0,
    Click     = 1,
    Bubble    = 2,
    Celestial = 3
};

class SoundEngine {
public:
    bool      enabled = true;
    float     volume  = 0.55f;
    SoundPack pack    = SoundPack::Click;

    void Init() {
        WAVEFORMATEX wfx{};
        wfx.wFormatTag      = WAVE_FORMAT_PCM;
        wfx.nChannels       = 1;
        wfx.nSamplesPerSec  = kSampleRate;
        wfx.wBitsPerSample  = 16;
        wfx.nBlockAlign     = 2;
        wfx.nAvgBytesPerSec = kSampleRate * 2;

        for (auto& v : voices_) {
            if (waveOutOpen(&v.hWave, WAVE_MAPPER, &wfx, 0, 0, CALLBACK_NULL) == MMSYSERR_NOERROR) {
                v.valid = true;
            }
        }
    }

    void Shutdown() {
        for (auto& v : voices_) {
            if (v.valid && v.hWave) {
                waveOutReset(v.hWave);
                if (v.prepared) {
                    waveOutUnprepareHeader(v.hWave, &v.hdr, sizeof(WAVEHDR));
                    v.prepared = false;
                }
                waveOutClose(v.hWave);
                v.hWave = nullptr;
                v.valid = false;
            }
        }
    }

    void PlayToggle(bool stateOn) {
        if (!enabled || volume <= 0.01f) return;
        switch (pack) {
            case SoundPack::Pop:
                TriggerTone(stateOn ? 520.0f : 390.0f, stateOn ? 780.0f : 270.0f, 0.065f, 0.9f);
                break;
            case SoundPack::Click:
                TriggerTone(stateOn ? 980.0f : 680.0f, stateOn ? 1320.0f : 440.0f, 0.042f, 0.8f);
                break;
            case SoundPack::Bubble:
                TriggerTone(stateOn ? 380.0f : 540.0f, stateOn ? 760.0f : 310.0f, 0.085f, 0.85f);
                break;
            case SoundPack::Celestial:
                TriggerChord(stateOn ? 587.33f : 523.25f, stateOn ? 880.0f : 392.0f, 0.14f, 0.75f);
                break;
        }
    }

    void PlayTab() {
        if (!enabled || volume <= 0.01f) return;
        TriggerTone(640.0f, 820.0f, 0.032f, 0.45f);
    }

    void PlayAlert() {
        if (!enabled || volume <= 0.01f) return;
        TriggerChord(783.99f, 1046.50f, 0.16f, 0.80f);
    }

    void PlayStartupChime() {
        if (!enabled || volume <= 0.01f) return;
        // Ascending tactical crystalline triad (D5 - F#5 - A5 - D6)
        TriggerArpeggio({ 587.33f, 739.99f, 880.00f, 1174.66f }, 0.36f, 0.72f);
    }

private:
    static constexpr int kSampleRate = 44100;
    static constexpr int kMaxSamples = 22050; // 0.5s max buffer per voice

    struct Voice {
        HWAVEOUT hWave = nullptr;
        WAVEHDR  hdr{};
        std::array<int16_t, kMaxSamples> pcm{};
        bool valid    = false;
        bool prepared = false;
    };

    std::array<Voice, 6> voices_{};
    size_t rrIndex_ = 0;

    Voice* AcquireVoice(int sampleCount) {
        for (size_t i = 0; i < voices_.size(); ++i) {
            Voice& v = voices_[(rrIndex_ + i) % voices_.size()];
            if (!v.valid) continue;
            if (v.prepared && !(v.hdr.dwFlags & WHDR_DONE)) continue;
            if (v.prepared) {
                waveOutUnprepareHeader(v.hWave, &v.hdr, sizeof(WAVEHDR));
                v.prepared = false;
            }
            rrIndex_ = (rrIndex_ + i + 1) % voices_.size();
            std::memset(&v.hdr, 0, sizeof(WAVEHDR));
            v.hdr.lpData = reinterpret_cast<LPSTR>(v.pcm.data());
            v.hdr.dwBufferLength = static_cast<DWORD>(sampleCount * sizeof(int16_t));
            return &v;
        }
        return nullptr;
    }

    void SubmitVoice(Voice* v) {
        if (!v) return;
        if (waveOutPrepareHeader(v->hWave, &v->hdr, sizeof(WAVEHDR)) == MMSYSERR_NOERROR) {
            v->prepared = true;
            waveOutWrite(v->hWave, &v->hdr, sizeof(WAVEHDR));
        }
    }

    void TriggerTone(float f0, float f1, float durSec, float ampScale) {
        int n = std::clamp(static_cast<int>(durSec * kSampleRate), 64, kMaxSamples);
        Voice* v = AcquireVoice(n);
        if (!v) return;
        float gain = std::clamp(volume * ampScale, 0.0f, 1.0f) * 14000.0f;
        float phase = 0.0f;
        for (int i = 0; i < n; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(n);
            float freq = f0 + (f1 - f0) * (1.0f - (1.0f - t) * (1.0f - t));
            phase += (2.0f * std::numbers::pi_v<float> * freq) / kSampleRate;
            float env = std::sin(std::min(1.0f, t * 14.0f) * 1.570796f) * std::exp(-t * 5.2f);
            float s = (std::sin(phase) + 0.22f * std::sin(phase * 2.0f)) * env * gain;
            v->pcm[i] = static_cast<int16_t>(std::clamp(s, -32000.0f, 32000.0f));
        }
        SubmitVoice(v);
    }

    void TriggerChord(float fA, float fB, float durSec, float ampScale) {
        int n = std::clamp(static_cast<int>(durSec * kSampleRate), 64, kMaxSamples);
        Voice* v = AcquireVoice(n);
        if (!v) return;
        float gain = std::clamp(volume * ampScale, 0.0f, 1.0f) * 10500.0f;
        for (int i = 0; i < n; ++i) {
            float sec = static_cast<float>(i) / kSampleRate;
            float t   = static_cast<float>(i) / static_cast<float>(n);
            float env = std::min(1.0f, t * 18.0f) * std::exp(-t * 4.4f);
            float s = (std::sin(2.0f * std::numbers::pi_v<float> * fA * sec) +
                       0.8f * std::sin(2.0f * std::numbers::pi_v<float> * fB * sec)) * env * gain;
            v->pcm[i] = static_cast<int16_t>(std::clamp(s, -32000.0f, 32000.0f));
        }
        SubmitVoice(v);
    }

    void TriggerArpeggio(std::initializer_list<float> notes, float durSec, float ampScale) {
        int n = std::clamp(static_cast<int>(durSec * kSampleRate), 256, kMaxSamples);
        Voice* v = AcquireVoice(n);
        if (!v) return;
        std::vector<float> freqs(notes);
        float gain = std::clamp(volume * ampScale, 0.0f, 1.0f) * 8500.0f;
        for (int i = 0; i < n; ++i) {
            float sec = static_cast<float>(i) / kSampleRate;
            float sum = 0.0f;
            for (size_t k = 0; k < freqs.size(); ++k) {
                float onset = static_cast<float>(k) * 0.055f;
                if (sec >= onset) {
                    float local = sec - onset;
                    float env = std::min(1.0f, local * 40.0f) * std::exp(-local * 5.8f);
                    sum += std::sin(2.0f * std::numbers::pi_v<float> * freqs[k] * local) * env;
                }
            }
            v->pcm[i] = static_cast<int16_t>(std::clamp(sum * gain, -32000.0f, 32000.0f));
        }
        SubmitVoice(v);
    }
};

static SoundEngine g_Sound;

// ============================================================================
// 4. WIC ANISOTROPIC BRAND LOGO & D3D11 HLSL SHADER RENDERER
// ============================================================================
class BrandLogoTexture {
public:
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11SamplerState*       anisoSampler = nullptr;
    int width  = 0;
    int height = 0;

    bool LoadFromPNG(ID3D11Device* dev, ID3D11DeviceContext* ctx, const fs::path& pngPath) {
        if (!dev || !ctx || !fs::exists(pngPath)) return false;

        IWICImagingFactory* factory = nullptr;
        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
        if (FAILED(hr) || !factory) return false;

        IWICBitmapDecoder* decoder = nullptr;
        hr = factory->CreateDecoderFromFilename(pngPath.wstring().c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder);
        if (FAILED(hr) || !decoder) { factory->Release(); return false; }

        IWICBitmapFrameDecode* frame = nullptr;
        hr = decoder->GetFrame(0, &frame);
        if (FAILED(hr) || !frame) { decoder->Release(); factory->Release(); return false; }

        IWICFormatConverter* conv = nullptr;
        hr = factory->CreateFormatConverter(&conv);
        if (FAILED(hr) || !conv) { frame->Release(); decoder->Release(); factory->Release(); return false; }

        hr = conv->Initialize(frame, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);
        if (FAILED(hr)) { conv->Release(); frame->Release(); decoder->Release(); factory->Release(); return false; }

        UINT w = 0, h = 0;
        conv->GetSize(&w, &h);
        if (w == 0 || h == 0) { conv->Release(); frame->Release(); decoder->Release(); factory->Release(); return false; }

        std::vector<uint8_t> rgba(w * h * 4);
        hr = conv->CopyPixels(nullptr, w * 4, static_cast<UINT>(rgba.size()), rgba.data());
        conv->Release();
        frame->Release();
        decoder->Release();
        factory->Release();
        if (FAILED(hr)) return false;

        // Premultiply RGB edges against alpha so bilinear/anisotropic mipmaps have zero dark halo
        for (size_t i = 0; i < rgba.size(); i += 4) {
            if (rgba[i + 3] == 0) {
                rgba[i + 0] = rgba[i + 1] = rgba[i + 2] = 0;
            }
        }

        D3D11_TEXTURE2D_DESC td{};
        td.Width          = w;
        td.Height         = h;
        td.MipLevels      = 0;
        td.ArraySize      = 1;
        td.Format         = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage          = D3D11_USAGE_DEFAULT;
        td.BindFlags      = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        td.MiscFlags      = D3D11_RESOURCE_MISC_GENERATE_MIPS;

        ID3D11Texture2D* tex = nullptr;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &tex)) || !tex) return false;

        ctx->UpdateSubresource(tex, 0, nullptr, rgba.data(), w * 4, 0);

        D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format                    = td.Format;
        sd.ViewDimension             = D3D11_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MostDetailedMip = 0;
        sd.Texture2D.MipLevels       = static_cast<UINT>(-1);

        hr = dev->CreateShaderResourceView(tex, &sd, &srv);
        tex->Release();
        if (FAILED(hr) || !srv) return false;

        ctx->GenerateMips(srv);

        D3D11_SAMPLER_DESC samp{};
        samp.Filter         = D3D11_FILTER_ANISOTROPIC;
        samp.AddressU       = D3D11_TEXTURE_ADDRESS_CLAMP;
        samp.AddressV       = D3D11_TEXTURE_ADDRESS_CLAMP;
        samp.AddressW       = D3D11_TEXTURE_ADDRESS_CLAMP;
        samp.MaxAnisotropy  = 16;
        samp.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
        samp.MaxLOD         = D3D11_FLOAT32_MAX;
        dev->CreateSamplerState(&samp, &anisoSampler);

        width  = static_cast<int>(w);
        height = static_cast<int>(h);
        return true;
    }

    void Shutdown() {
        if (anisoSampler) { anisoSampler->Release(); anisoSampler = nullptr; }
        if (srv) { srv->Release(); srv = nullptr; }
    }
};

static BrandLogoTexture g_BrandLogo;

struct ShaderParamsCB {
    float rectMin[2];
    float rectMax[2];
    float screenSize[2];
    float time;
    float alpha;
    float accentA[4];
    float accentB[4];
    float bgBase[4];
    float rounding;
    float sheenStrength;
    float glowStrength;
    int   mode;
};

class MenuShaderRenderer {
public:
    bool      backdropEnabled    = true;
    bool      screenAtmosEnabled = false;
    int       screenAtmosPreset  = 0; // 0=Tactical Vignette, 1=CRT Scanlines, 2=Laser Grid, 3=Crimson Nebula
    float     screenAtmosAlpha   = 0.30f;
    float     sheenIntensity     = 0.55f;
    float     glowIntensity      = 0.65f;

    ID3D11VertexShader*    vs          = nullptr;
    ID3D11PixelShader*     psBackdrop  = nullptr;
    ID3D11PixelShader*     psAtmos     = nullptr;
    ID3D11PixelShader*     psLoading   = nullptr;
    ID3D11Buffer*          cbuffer     = nullptr;
    ID3D11BlendState*      blendState  = nullptr;
    ID3D11RasterizerState* noScissorRS = nullptr;
    ID3D11DeviceContext*   cachedCtx   = nullptr;
    ID3D11RenderTargetView* cachedRtv  = nullptr;
    ShaderParamsCB         pendingBackdrop{};
    bool                   ready       = false;

    bool Init(ID3D11Device* dev) {
        if (!dev) return false;

        static const char* kFullscreenVS = R"(
            struct VSOut {
                float4 pos : SV_POSITION;
                float2 uv  : TEXCOORD0;
            };
            VSOut main(uint vid : SV_VertexID) {
                VSOut o;
                float2 uv = float2((vid << 1) & 2, vid & 2);
                o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
                o.uv  = uv;
                return o;
            }
        )";

        static const char* kBackdropPS = R"(
            cbuffer Params : register(b0) {
                float2 rectMin;
                float2 rectMax;
                float2 screenSize;
                float  time;
                float  alpha;
                float4 accentA;
                float4 accentB;
                float4 bgBase;
                float  rounding;
                float  sheenStrength;
                float  glowStrength;
                int    mode;
            };
            float sdRoundBox(float2 p, float2 b, float r) {
                float2 q = abs(p) - b + r;
                return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
            }
            float hash21(float2 p) {
                p = frac(p * float2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return frac(p.x * p.y);
            }
            float4 main(float4 svPos : SV_POSITION) : SV_TARGET {
                float2 px = svPos.xy;
                float2 center = 0.5 * (rectMin + rectMax);
                float2 halfSz = 0.5 * (rectMax - rectMin);
                float d = sdRoundBox(px - center, halfSz, rounding);
                if (d > 1.5) discard;
                float edgeMask = saturate(1.0 - smoothstep(-1.0, 1.0, d));

                float2 uv = (px - rectMin) / max(rectMax - rectMin, float2(1.0, 1.0));
                float3 col = bgBase.rgb;

                float2 p1 = float2(0.14 + 0.06 * sin(time * 0.55), 0.16 + 0.05 * cos(time * 0.45));
                float2 p2 = float2(0.86 + 0.05 * cos(time * 0.48), 0.82 + 0.06 * sin(time * 0.62));
                float g1 = exp(-dot(uv - p1, uv - p1) * 6.5);
                float g2 = exp(-dot(uv - p2, uv - p2) * 5.8);
                col += accentA.rgb * (g1 * 0.11 * glowStrength);
                col += accentB.rgb * (g2 * 0.09 * glowStrength);

                float diag = uv.x * 0.72 + uv.y * 0.68;
                float sweepPos = frac(time * 0.085) * 2.2 - 0.5;
                float sheen = exp(-pow((diag - sweepPos) * 5.2, 2.0));
                col += lerp(accentA.rgb, float3(1.0, 1.0, 1.0), 0.55) * (sheen * 0.055 * sheenStrength);

                float vig = saturate(1.0 - 0.22 * dot(uv - 0.5, uv - 0.5) * 2.2);
                col *= lerp(1.0, vig, 0.45);
                float grain = (hash21(px + frac(time) * 100.0) - 0.5) * (1.5 / 255.0);
                col += grain;

                return float4(saturate(col), alpha * edgeMask);
            }
        )";

        static const char* kAtmosPS = R"(
            cbuffer Params : register(b0) {
                float2 rectMin;
                float2 rectMax;
                float2 screenSize;
                float  time;
                float  alpha;
                float4 accentA;
                float4 accentB;
                float4 bgBase;
                float  rounding;
                float  sheenStrength;
                float  glowStrength;
                int    mode;
            };
            float4 main(float4 svPos : SV_POSITION) : SV_TARGET {
                float2 uv = svPos.xy / max(screenSize, float2(1.0, 1.0));
                float2 c  = uv - 0.5;
                float3 col = float3(0.02, 0.02, 0.03);
                float outA = alpha;

                if (mode == 0) {
                    float r2 = dot(c, c) * 2.4;
                    col = lerp(float3(0.03, 0.03, 0.045), accentA.rgb * 0.16, saturate(r2 * 0.65));
                    outA = alpha * saturate(0.35 + r2 * 0.85);
                } else if (mode == 1) {
                    float scan = 0.5 + 0.5 * sin(svPos.y * 3.14159);
                    float roll = exp(-pow((uv.y - frac(time * 0.14)) * 9.0, 2.0));
                    col = accentA.rgb * (0.06 + 0.12 * roll) + float3(0.01, 0.01, 0.015) * scan;
                    outA = alpha * (0.28 + 0.35 * scan + 0.25 * roll);
                } else if (mode == 2) {
                    float2 g = abs(frac(svPos.xy / 44.0 + float2(time * 0.04, time * 0.02)) - 0.5);
                    float lineMask = 1.0 - smoothstep(0.0, 0.045, min(g.x, g.y));
                    col = lerp(float3(0.02, 0.02, 0.03), accentA.rgb, lineMask * 0.55);
                    outA = alpha * (0.20 + lineMask * 0.45);
                } else {
                    float w1 = sin(uv.x * 4.5 + time * 0.7) * cos(uv.y * 3.8 - time * 0.5);
                    float w2 = cos((uv.x + uv.y) * 5.2 - time * 0.6);
                    col = lerp(accentA.rgb, accentB.rgb, saturate(0.5 + 0.5 * w1)) * (0.18 + 0.10 * w2);
                    outA = alpha * 0.52;
                }
                return float4(saturate(col), saturate(outA));
            }
        )";

        static const char* kLoadingSkyPS = R"(
            cbuffer Params : register(b0) {
                float2 rectMin;
                float2 rectMax;
                float2 screenSize;
                float  time;
                float  alpha;
                float4 accentA;
                float4 accentB;
                float4 bgBase;
                float  rounding;
                float  sheenStrength;
                float  glowStrength;
                int    mode;
            };
            float hash21(float2 p) {
                p = frac(p * float2(234.34, 435.345));
                p += dot(p, p + 34.23);
                return frac(p.x * p.y);
            }
            float noise2(float2 p) {
                float2 i = floor(p);
                float2 f = frac(p);
                f = f * f * (3.0 - 2.0 * f);
                float a = hash21(i);
                float b = hash21(i + float2(1.0, 0.0));
                float c = hash21(i + float2(0.0, 1.0));
                float d = hash21(i + float2(1.0, 1.0));
                return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
            }
            float fbm(float2 p) {
                float v = 0.0, amp = 0.5;
                for (int k = 0; k < 4; ++k) {
                    v += amp * noise2(p);
                    p *= 2.03;
                    amp *= 0.5;
                }
                return v;
            }
            float4 main(float4 svPos : SV_POSITION) : SV_TARGET {
                float2 uv = svPos.xy / max(screenSize, float2(1.0, 1.0));
                float3 zenith  = float3(0.030, 0.032, 0.052);
                float3 midSky  = float3(0.058, 0.048, 0.088);
                float3 horizon = float3(0.145, 0.062, 0.115);
                float3 col = lerp(zenith, midSky, smoothstep(0.0, 0.62, uv.y));
                col = lerp(col, horizon, smoothstep(0.48, 1.0, uv.y));

                float cloud = fbm(uv * float2(3.6, 2.2) + float2(time * 0.045, -time * 0.02));
                col += accentA.rgb * smoothstep(0.38, 0.85, cloud) * 0.14;
                col += accentB.rgb * exp(-dot(uv - float2(0.5, 0.44), uv - float2(0.5, 0.44)) * 5.2) * 0.16;

                // Subtle tactical perspective floor grid
                if (uv.y > 0.62) {
                    float py = (uv.y - 0.62) / 0.38;
                    float gx = abs(frac((uv.x - 0.5) / max(py, 0.06) * 0.38) - 0.5);
                    float gy = abs(frac(1.2 / max(py, 0.05) - time * 0.45) - 0.5);
                    float grid = (1.0 - smoothstep(0.0, 0.035, gx)) + (1.0 - smoothstep(0.0, 0.045, gy));
                    col += accentA.rgb * saturate(grid) * py * 0.11;
                }

                // Crisp starfield / ember dust
                float2 cell = floor(svPos.xy / 38.0);
                float2 local = frac(svPos.xy / 38.0) - 0.5;
                float rnd = hash21(cell);
                if (rnd > 0.84) {
                    float2 sp = (float2(hash21(cell + 7.1), hash21(cell + 13.7)) - 0.5) * 0.55;
                    float dist = length(local - sp);
                    float tw = 0.5 + 0.5 * sin(time * 3.4 + rnd * 30.0);
                    col += lerp(accentA.rgb, float3(1.0, 0.92, 0.95), 0.6) * exp(-dist * 24.0) * tw * 0.45;
                }

                float vig = 1.0 - 0.35 * dot(uv - 0.5, uv - 0.5) * 1.8;
                return float4(saturate(col * vig), alpha);
            }
        )";

        auto compileShader = [](const char* src, const char* target, ID3DBlob** outBlob) -> bool {
            ID3DBlob* err = nullptr;
            HRESULT hr = D3DCompile(src, std::strlen(src), nullptr, nullptr, nullptr, "main", target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, outBlob, &err);
            if (err) err->Release();
            return SUCCEEDED(hr) && *outBlob != nullptr;
        };

        ID3DBlob* bVS = nullptr;
        ID3DBlob* bBack = nullptr;
        ID3DBlob* bAtmos = nullptr;
        ID3DBlob* bLoad = nullptr;
        if (!compileShader(kFullscreenVS, "vs_5_0", &bVS) ||
            !compileShader(kBackdropPS,   "ps_5_0", &bBack) ||
            !compileShader(kAtmosPS,      "ps_5_0", &bAtmos) ||
            !compileShader(kLoadingSkyPS, "ps_5_0", &bLoad)) {
            if (bVS) bVS->Release();
            if (bBack) bBack->Release();
            if (bAtmos) bAtmos->Release();
            if (bLoad) bLoad->Release();
            return false;
        }

        dev->CreateVertexShader(bVS->GetBufferPointer(), bVS->GetBufferSize(), nullptr, &vs);
        dev->CreatePixelShader(bBack->GetBufferPointer(), bBack->GetBufferSize(), nullptr, &psBackdrop);
        dev->CreatePixelShader(bAtmos->GetBufferPointer(), bAtmos->GetBufferSize(), nullptr, &psAtmos);
        dev->CreatePixelShader(bLoad->GetBufferPointer(), bLoad->GetBufferSize(), nullptr, &psLoading);
        bVS->Release();
        bBack->Release();
        bAtmos->Release();
        bLoad->Release();

        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth      = sizeof(ShaderParamsCB);
        bd.Usage          = D3D11_USAGE_DYNAMIC;
        bd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        dev->CreateBuffer(&bd, nullptr, &cbuffer);

        D3D11_BLEND_DESC blend{};
        blend.RenderTarget[0].BlendEnable           = TRUE;
        blend.RenderTarget[0].SrcBlend              = D3D11_BLEND_SRC_ALPHA;
        blend.RenderTarget[0].DestBlend             = D3D11_BLEND_INV_SRC_ALPHA;
        blend.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;
        blend.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ONE;
        blend.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_INV_SRC_ALPHA;
        blend.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;
        blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        dev->CreateBlendState(&blend, &blendState);

        D3D11_RASTERIZER_DESC rs{};
        rs.FillMode        = D3D11_FILL_SOLID;
        rs.CullMode        = D3D11_CULL_NONE;
        rs.DepthClipEnable = TRUE;
        rs.ScissorEnable   = FALSE;
        dev->CreateRasterizerState(&rs, &noScissorRS);

        ready = (vs && psBackdrop && psAtmos && psLoading && cbuffer && blendState && noScissorRS);
        return ready;
    }

    void DrawFullscreenPass(ID3D11DeviceContext* ctx, ID3D11RenderTargetView* rtv, ID3D11PixelShader* ps, const ShaderParamsCB& params) {
        if (!ready || !ctx || !rtv || !ps) return;

        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (SUCCEEDED(ctx->Map(cbuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            std::memcpy(mapped.pData, &params, sizeof(ShaderParamsCB));
            ctx->Unmap(cbuffer, 0);
        }

        D3D11_VIEWPORT vp{};
        vp.Width    = params.screenSize[0];
        vp.Height   = params.screenSize[1];
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;
        ctx->RSSetState(noScissorRS);
        ctx->RSSetViewports(1, &vp);
        ctx->OMSetRenderTargets(1, &rtv, nullptr);

        float blendFactor[4] = { 0, 0, 0, 0 };
        ctx->OMSetBlendState(blendState, blendFactor, 0xFFFFFFFF);
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->VSSetShader(vs, nullptr, 0);
        ctx->PSSetShader(ps, nullptr, 0);
        ctx->PSSetConstantBuffers(0, 1, &cbuffer);
        ctx->Draw(3, 0);
    }

    static void BackdropDrawCallback(const ImDrawList*, const ImDrawCmd* cmd) {
        auto* self = static_cast<MenuShaderRenderer*>(cmd->UserCallbackData);
        if (!self || !self->ready || !self->cachedCtx || !self->cachedRtv) return;
        self->DrawFullscreenPass(self->cachedCtx, self->cachedRtv, self->psBackdrop, self->pendingBackdrop);
    }

    void Shutdown() {
        if (noScissorRS) { noScissorRS->Release(); noScissorRS = nullptr; }
        if (blendState)  { blendState->Release();  blendState  = nullptr; }
        if (cbuffer)    { cbuffer->Release();    cbuffer    = nullptr; }
        if (psLoading)  { psLoading->Release();  psLoading  = nullptr; }
        if (psAtmos)    { psAtmos->Release();    psAtmos    = nullptr; }
        if (psBackdrop) { psBackdrop->Release(); psBackdrop = nullptr; }
        if (vs)         { vs->Release();         vs         = nullptr; }
        ready = false;
    }
};

static MenuShaderRenderer g_Shaders;

// ============================================================================
// 5. PROCESS RAII & COMBAT MASTER REORDERED-IL2CPP + UNITY WORLD SCANNER
// ============================================================================
class ScopedHandle {
public:
    ScopedHandle() = default;
    explicit ScopedHandle(HANDLE h) : h_(h) {}
    ~ScopedHandle() { Reset(); }
    ScopedHandle(const ScopedHandle&) = delete;
    ScopedHandle& operator=(const ScopedHandle&) = delete;
    ScopedHandle(ScopedHandle&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    ScopedHandle& operator=(ScopedHandle&& o) noexcept {
        if (this != &o) { Reset(); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    void Reset(HANDLE nh = nullptr) {
        if (h_ && h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
        h_ = nh;
    }
    HANDLE Get() const { return h_; }
    bool Valid() const { return h_ != nullptr && h_ != INVALID_HANDLE_VALUE; }
private:
    HANDLE h_ = nullptr;
};

struct BoneSkeleton3D {
    Vec3 head{};
    Vec3 neck{};
    Vec3 chest{};
    Vec3 pelvis{};
    Vec3 lKnee{};
    Vec3 rKnee{};
    Vec3 lFoot{};
    Vec3 rFoot{};
    Vec3 rHand{};
    Vec3 lHand{};
    bool hasRealBones = false;
};

struct OperatorEntity {
    uintptr_t      address      = 0;
    Vec3           feetPos{};
    Vec3           headPos{};
    Vec3           velocity{};
    BoneSkeleton3D bones{};
    float          health       = 100.0f;
    float          maxHealth    = 100.0f;
    float          armor        = 0.0f;
    float          maxArmor     = 100.0f;
    int            teamId       = -1;
    bool           isTeammate   = false;
    bool           isBot        = false;
    bool           isDowned     = false;
    bool           isCrouched   = false;
    float          distanceM    = 0.0f;
    std::string    name         = "Operator";
    std::string    weaponName   = "Rifle";
};

struct PlantedBombInfo {
    bool  active    = false;
    Vec3  position{};
    float timeLeft  = 0.0f;
    float maxTime   = 40.0f;
};

struct WorldSnapshot {
    bool                       attached       = false;
    DWORD                      pid            = 0;
    HWND                       gameHwnd       = nullptr;
    bool                       inMatch        = false;
    bool                       cursorLocked   = false;
    Mat4                       vpMatrix{};
    Vec3                       camPos{};
    Vec3                       camFwd{ 0.0f, 0.0f, 1.0f };
    Vec3                       camRight{ 1.0f, 0.0f, 0.0f };
    Vec3                       camUp{ 0.0f, 1.0f, 0.0f };
    float                      camFovDeg      = 90.0f;
    uintptr_t                  localPlayerPtr = 0;
    Vec3                       localPos{};
    Vec3                       localVel{};
    int                        localTeam      = -1;
    std::vector<OperatorEntity> players{};
    PlantedBombInfo            bomb{};
    uint64_t                   rpmPerSec      = 0;
    uint64_t                   seq            = 0;
};

class CombatMasterScanner {
public:
    // Verified Binary RVAs from REA & dumpbin analysis of UnityPlayer.dll & GameAssembly.dll
    static constexpr uintptr_t kUP_GameObjectManager      = 0x2232770;
    static constexpr uintptr_t kUP_CameraManager          = 0x2232838;
    static constexpr uintptr_t kUP_CursorManager          = 0x22328F8;
    static constexpr uintptr_t kUP_CameraTypeBase         = 0x21170C0;
    static constexpr uintptr_t kUP_CameraTypeCount        = 0x21170C4;

    static constexpr uintptr_t kGA_TypeInfoDefTable       = 0x4C21200; // Il2CppClass*[22103]
    static constexpr uintptr_t kGA_PlayerRoot_TypeInfo    = 0x49B5BC0; // CombatMaster.Battle.Gameplay.Player.PlayerRoot

    // Reordered Combat Master Il2CppClass & FieldInfo offsets (verified in 0x18048C090 & 0x1804D2356)
    static constexpr uintptr_t kIl2CppClass_Namespaze     = 0x18;
    static constexpr uintptr_t kIl2CppClass_Parent        = 0x78;
    static constexpr uintptr_t kIl2CppClass_Fields        = 0x90;
    static constexpr uintptr_t kIl2CppClass_Name          = 0xA0;
    static constexpr uintptr_t kIl2CppClass_StaticFields  = 0xB8;
    static constexpr uintptr_t kIl2CppClass_FieldCount    = 0x124; // uint16_t

    static constexpr uintptr_t kFieldInfo_Stride          = 0x28;
    static constexpr uintptr_t kFieldInfo_Name            = 0x10;
    static constexpr uintptr_t kFieldInfo_Offset          = 0x20;

    WorldSnapshot GetSnapshot() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return snapshot_;
    }

    void Start() {
        if (running_.exchange(true)) return;
        worker_ = std::thread([this]() { WorkerLoop(); });
    }

    void Stop() {
        running_.store(false);
        if (worker_.joinable()) worker_.join();
        proc_.Reset();
    }

private:
    struct PrevMotionSample {
        Vec3   pos{};
        double timestamp = 0.0;
        Vec3   smoothedVel{};
    };

    struct ResolvedOffsets {
        bool reflectionScanned = false;
        // PlayerRoot instance offsets (verified in CombatMaster GameAssembly.dll)
        int32_t pr_state            = 0x28;
        int32_t pr_firstPivot       = 0x30;
        int32_t pr_thirdPivot       = 0x38;
        int32_t pr_camController    = 0x48;
        int32_t pr_firstView        = 0x80;
        int32_t pr_thirdView        = 0x88;
        int32_t pr_collider         = 0x98;
        int32_t pr_charCtrl         = 0xA0;
        int32_t pr_arming           = 0xA8;
        int32_t pr_movement         = 0xB0;
        int32_t pr_health           = 0xB8;
        int32_t pr_isOwner          = 0x108;
        int32_t pr_hasControl       = 0x109;
        int32_t pr_isVisible        = 0x10A;
        int32_t pr_isFirstPerson    = 0x131;
        int32_t pr_isRealPlayer     = 0x132;
        int32_t pr_cachedPlayerData = 0x150;
        int32_t pr_info             = 0x1E8;
        int32_t pr_matchInfo        = 0x1F8;
        int32_t pr_isInit           = 0x290;

        // Battle CameraController instance offsets
        int32_t cc_renderCamera     = 0x210;

        // PlayerHealth instance offsets
        int32_t ph_lastArmor        = 0xCC;
        int32_t ph_lastHealth       = 0xD0;
        int32_t ph_deathCounter     = 0xB0;
        int32_t ph_selfReviveEnd    = 0xA4;

        // PlayerMovement instance offsets
        int32_t pm_velocity         = 0x80;
        int32_t pm_isCrouch         = 0x90;

        // PlayerArming instance offsets
        int32_t pa_primaryWeapon    = 0xD0;
        int32_t pa_mainSlotWeapon   = 0x130;
        int32_t pa_activeWeapon     = 0x538;

        // BaseGameModeSystem, ActorPlantedBomb & SNDGameModeSystem
        uintptr_t baseGmKlass       = 0;
        uintptr_t plantedBombKlass  = 0;
        uintptr_t sndSystemKlass    = 0;
        int32_t   bomb_timeLeft     = 0x28;
        int32_t   snd_bombPos       = 0xF0;
        int32_t   snd_bombGo        = 0x108;
    };

    std::atomic<bool> running_{ false };
    std::thread       worker_;
    mutable std::mutex mtx_;
    WorldSnapshot     snapshot_{};

    ScopedHandle      proc_;
    DWORD             pid_              = 0;
    HWND              hwnd_             = nullptr;
    uintptr_t         unityPlayerBase_  = 0;
    uintptr_t         gameAssemblyBase_ = 0;
    uint64_t          rpmCounter_       = 0;
    uint64_t          rpmLastSec_       = 0;
    uint64_t          snapSeq_          = 0;
    double            lastRpmStamp_     = 0.0;
    double            lastReflectStamp_ = 0.0;
    ResolvedOffsets   off_{};
    std::unordered_map<uintptr_t, PrevMotionSample> motionCache_;

    static bool IsValidPtr(uintptr_t p) {
        return p >= 0x10000ULL && p < 0x00007FFFFFFFFFFFULL;
    }

    bool ReadRaw(uintptr_t addr, void* buf, size_t sz) {
        if (!proc_.Valid() || !IsValidPtr(addr) || sz == 0) return false;
        SIZE_T bytesRead = 0;
        ++rpmCounter_;
        return ReadProcessMemory(proc_.Get(), reinterpret_cast<LPCVOID>(addr), buf, sz, &bytesRead) && bytesRead == sz;
    }

    template <typename T>
    bool Read(uintptr_t addr, T& out) {
        return ReadRaw(addr, &out, sizeof(T));
    }

    template <typename T>
    T ReadOr(uintptr_t addr, T fallback = T{}) {
        T v{};
        return Read(addr, v) ? v : fallback;
    }

    std::string ReadAscii(uintptr_t addr, size_t maxLen = 64) {
        if (!IsValidPtr(addr)) return {};
        char buf[96]{};
        size_t toRead = std::min(maxLen, sizeof(buf) - 1);
        SIZE_T n = 0;
        ++rpmCounter_;
        if (!ReadProcessMemory(proc_.Get(), reinterpret_cast<LPCVOID>(addr), buf, toRead, &n) || n == 0) {
            return {};
        }
        buf[n] = '\0';
        std::string s;
        s.reserve(n);
        for (size_t i = 0; i < n && buf[i] != '\0'; ++i) {
            unsigned char c = static_cast<unsigned char>(buf[i]);
            if (c < 0x20 || c > 0x7E) return {};
            s.push_back(buf[i]);
        }
        return s;
    }

    std::string ReadIl2CppString(uintptr_t strObj, int maxChars = 36) {
        if (!IsValidPtr(strObj)) return {};
        int32_t len = 0;
        if (!Read(strObj + 0x10, len) || len <= 0 || len > maxChars) return {};
        std::array<wchar_t, 40> wbuf{};
        int count = std::min(len, 38);
        if (!ReadRaw(strObj + 0x14, wbuf.data(), static_cast<size_t>(count) * sizeof(wchar_t))) return {};
        wbuf[count] = L'\0';

        // Validate printable UTF-16 code units
        for (int i = 0; i < count; ++i) {
            if (wbuf[i] < 0x20 && wbuf[i] != L'\t') return {};
        }
        int needed = WideCharToMultiByte(CP_UTF8, 0, wbuf.data(), count, nullptr, 0, nullptr, nullptr);
        if (needed <= 0 || needed > 120) return {};
        std::string out(static_cast<size_t>(needed), '\0');
        WideCharToMultiByte(CP_UTF8, 0, wbuf.data(), count, out.data(), needed, nullptr, nullptr);
        return out;
    }

    static DWORD FindProcessId(const wchar_t* exeName) {
        ScopedHandle snap(CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0));
        if (!snap.Valid()) return 0;
        PROCESSENTRY32W pe{};
        pe.dwSize = sizeof(pe);
        if (Process32FirstW(snap.Get(), &pe)) {
            do {
                if (_wcsicmp(pe.szExeFile, exeName) == 0) return pe.th32ProcessID;
            } while (Process32NextW(snap.Get(), &pe));
        }
        return 0;
    }

    static uintptr_t FindModuleBase(DWORD pid, const wchar_t* modName) {
        ScopedHandle snap(CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid));
        if (!snap.Valid()) return 0;
        MODULEENTRY32W me{};
        me.dwSize = sizeof(me);
        if (Module32FirstW(snap.Get(), &me)) {
            do {
                if (_wcsicmp(me.szModule, modName) == 0) {
                    return reinterpret_cast<uintptr_t>(me.modBaseAddr);
                }
            } while (Module32NextW(snap.Get(), &me));
        }
        return 0;
    }

    static HWND FindUnityWindowForPid(DWORD targetPid) {
        struct EnumCtx { DWORD pid; HWND found; } ctx{ targetPid, nullptr };
        EnumWindows([](HWND hWnd, LPARAM lParam) -> BOOL {
            auto* c = reinterpret_cast<EnumCtx*>(lParam);
            DWORD wndPid = 0;
            GetWindowThreadProcessId(hWnd, &wndPid);
            if (wndPid != c->pid || !IsWindowVisible(hWnd)) return TRUE;
            wchar_t cls[64]{};
            GetClassNameW(hWnd, cls, 63);
            if (wcsstr(cls, L"UnityWndClass") != nullptr) {
                c->found = hWnd;
                return FALSE;
            }
            if (!c->found) c->found = hWnd;
            return TRUE;
        }, reinterpret_cast<LPARAM>(&ctx));
        return ctx.found;
    }

    bool TryAttach() {
        DWORD pid = FindProcessId(L"CombatMaster.exe");
        if (pid == 0) return false;

        ScopedHandle h(OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid));
        if (!h.Valid()) return false;

        // Bitness verification (non-negotiable per core.md)
        BOOL isWow64 = FALSE;
        if (!IsWow64Process(h.Get(), &isWow64) || isWow64 != FALSE) {
            return false;
        }

        uintptr_t up = FindModuleBase(pid, L"UnityPlayer.dll");
        uintptr_t ga = FindModuleBase(pid, L"GameAssembly.dll");
        if (!up || !ga) return false;

        proc_             = std::move(h);
        pid_              = pid;
        unityPlayerBase_  = up;
        gameAssemblyBase_ = ga;
        hwnd_             = FindUnityWindowForPid(pid);
        off_              = ResolvedOffsets{};
        motionCache_.clear();
        return true;
    }

    void ReadClassFields(uintptr_t klass, std::unordered_map<std::string, int32_t>& outMap) {
        uintptr_t cur = klass;
        for (int depth = 0; depth < 6 && IsValidPtr(cur); ++depth) {
            uint16_t fcount = ReadOr<uint16_t>(cur + kIl2CppClass_FieldCount, 0);
            uintptr_t fieldsPtr = ReadOr<uintptr_t>(cur + kIl2CppClass_Fields, 0);
            if (fcount > 0 && fcount < 256 && IsValidPtr(fieldsPtr)) {
                std::vector<uint8_t> raw(static_cast<size_t>(fcount) * kFieldInfo_Stride);
                if (ReadRaw(fieldsPtr, raw.data(), raw.size())) {
                    for (uint16_t i = 0; i < fcount; ++i) {
                        const uint8_t* entry = raw.data() + static_cast<size_t>(i) * kFieldInfo_Stride;
                        uintptr_t namePtr = *reinterpret_cast<const uintptr_t*>(entry + kFieldInfo_Name);
                        int32_t offset    = *reinterpret_cast<const int32_t*>(entry + kFieldInfo_Offset);
                        if (offset >= 0 && offset < 0x2000 && IsValidPtr(namePtr)) {
                            std::string fname = ReadAscii(namePtr, 60);
                            if (!fname.empty() && outMap.find(fname) == outMap.end()) {
                                outMap[fname] = offset;
                            }
                        }
                    }
                }
            }
            cur = ReadOr<uintptr_t>(cur + kIl2CppClass_Parent, 0);
        }
    }

    void RefreshIl2CppReflection(double nowSec) {
        if (off_.reflectionScanned && (nowSec - lastReflectStamp_) < 5.0) return;
        lastReflectStamp_ = nowSec;

        uintptr_t prKlass = ReadOr<uintptr_t>(gameAssemblyBase_ + kGA_PlayerRoot_TypeInfo, 0);
        if (IsValidPtr(prKlass) && (prKlass & 1) == 0 && prKlass > 0x100000000ULL) {
            std::unordered_map<std::string, int32_t> prFields;
            ReadClassFields(prKlass, prFields);
            auto grab = [&](const char* k, int32_t& dst) {
                auto it = prFields.find(k);
                if (it != prFields.end() && it->second >= 0x10) dst = it->second;
            };
            grab("_state",                                off_.pr_state);
            grab("_firstPersonPivot",                     off_.pr_firstPivot);
            grab("_thirdPersonPivot",                     off_.pr_thirdPivot);
            grab("_cameraController",                     off_.pr_camController);
            grab("_firstPersonView",                      off_.pr_firstView);
            grab("_thirdPersonView",                      off_.pr_thirdView);
            grab("_playerCollider",                       off_.pr_collider);
            grab("_characterController",                  off_.pr_charCtrl);
            grab("_playerArming",                         off_.pr_arming);
            grab("_playerMovement",                       off_.pr_movement);
            grab("_playerHealth",                         off_.pr_health);
            grab("IsOwner",                               off_.pr_isOwner);
            grab("HasControl",                            off_.pr_hasControl);
            grab("_isVisible",                            off_.pr_isVisible);
            grab("<IsFirstPersonState>k__BackingField",   off_.pr_isFirstPerson);
            grab("<IsRealPlayer>k__BackingField",         off_.pr_isRealPlayer);
            grab("_cachedPlayerData",                     off_.pr_cachedPlayerData);
            grab("<Info>k__BackingField",                 off_.pr_info);
            grab("<MatchInfo>k__BackingField",            off_.pr_matchInfo);
            grab("<IsInitialized>k__BackingField",        off_.pr_isInit);
        }

        // Scan s_TypeInfoDefinitionTable (22,103 pointers = 176 KB single RPM) for component & game mode classes
        uintptr_t defTbl = ReadOr<uintptr_t>(gameAssemblyBase_ + kGA_TypeInfoDefTable, 0);
        if (!IsValidPtr(defTbl)) return;

        std::vector<uintptr_t> klasses(22103, 0);
        if (!ReadRaw(defTbl, klasses.data(), klasses.size() * sizeof(uintptr_t))) return;

        int matched = 0;
        for (uintptr_t kl : klasses) {
            if (!IsValidPtr(kl) || (kl & 7) != 0) continue;
            uintptr_t namePtr = ReadOr<uintptr_t>(kl + kIl2CppClass_Name, 0);
            if (!IsValidPtr(namePtr)) continue;
            std::string cname = ReadAscii(namePtr, 40);
            if (cname.empty()) continue;

            if (cname == "PlayerHealth") {
                std::unordered_map<std::string, int32_t> f;
                ReadClassFields(kl, f);
                if (f.count("_lastKnowHealth"))  off_.ph_lastHealth   = f["_lastKnowHealth"];
                if (f.count("_lastKnowArmor"))   off_.ph_lastArmor    = f["_lastKnowArmor"];
                if (f.count("_deathCounter"))    off_.ph_deathCounter = f["_deathCounter"];
                ++matched;
            } else if (cname == "PlayerMovement") {
                std::unordered_map<std::string, int32_t> f;
                ReadClassFields(kl, f);
                if (f.count("<Velocity>k__BackingField")) off_.pm_velocity = f["<Velocity>k__BackingField"];
                if (f.count("_isCrouch"))                 off_.pm_isCrouch = f["_isCrouch"];
                ++matched;
            } else if (cname == "PlayerArming") {
                std::unordered_map<std::string, int32_t> f;
                ReadClassFields(kl, f);
                if (f.count("_activeWeapon"))                        off_.pa_activeWeapon   = f["_activeWeapon"];
                if (f.count("<MainSlotsActiveWeapon>k__BackingField")) off_.pa_mainSlotWeapon = f["<MainSlotsActiveWeapon>k__BackingField"];
                if (f.count("_primaryWeapon"))                       off_.pa_primaryWeapon  = f["_primaryWeapon"];
                ++matched;
            } else if (cname == "CameraController") {
                std::string ns = ReadAscii(ReadOr<uintptr_t>(kl + kIl2CppClass_Namespaze, 0), 60);
                if (ns.find("Gameplay") != std::string::npos) {
                    std::unordered_map<std::string, int32_t> f;
                    ReadClassFields(kl, f);
                    if (f.count("_renderCamera")) off_.cc_renderCamera = f["_renderCamera"];
                    ++matched;
                }
            } else if (cname == "BaseGameModeSystem") {
                uint16_t fc = ReadOr<uint16_t>(kl + kIl2CppClass_FieldCount, 0);
                if (fc == 3) {
                    off_.baseGmKlass = kl;
                    ++matched;
                }
            } else if (cname == "ActorPlantedBomb") {
                off_.plantedBombKlass = kl;
                std::unordered_map<std::string, int32_t> f;
                ReadClassFields(kl, f);
                if (f.count("_timeLeft")) off_.bomb_timeLeft = f["_timeLeft"];
                ++matched;
            } else if (cname == "SNDGameModeSystem") {
                off_.sndSystemKlass = kl;
                std::unordered_map<std::string, int32_t> f;
                ReadClassFields(kl, f);
                if (f.count("_bombPosition"))   off_.snd_bombPos = f["_bombPosition"];
                if (f.count("_bombGameObject")) off_.snd_bombGo  = f["_bombGameObject"];
                ++matched;
            }
        }
        if (matched >= 2) off_.reflectionScanned = true;
    }

    // Exact Unity TransformHierarchy world-space solver verified in UnityPlayer.dll 0x180059AF0
    bool SolveNativeTransformWorldPos(uintptr_t nativeTransform, Vec3& outPos) {
        if (!IsValidPtr(nativeTransform)) return false;
        struct TransformAccess {
            uintptr_t hierarchy;
            int32_t   index;
        } ta{};
        if (!Read(nativeTransform + 0x28, ta) || !IsValidPtr(ta.hierarchy) || ta.index < 0 || ta.index > 32768) {
            return false;
        }

        uintptr_t trsArray      = ReadOr<uintptr_t>(ta.hierarchy + 0x18, 0);
        uintptr_t parentIndices = ReadOr<uintptr_t>(ta.hierarchy + 0x20, 0);
        if (!IsValidPtr(trsArray) || !IsValidPtr(parentIndices)) return false;

        struct alignas(16) TrsX {
            float px, py, pz, pw;
            float rx, ry, rz, rw;
            float sx, sy, sz, sw;
        };

        TrsX curTrs{};
        if (!Read(trsArray + static_cast<uintptr_t>(ta.index) * 0x30, curTrs)) return false;
        Vec3 result{ curTrs.px, curTrs.py, curTrs.pz };
        if (!result.IsFinite()) return false;

        int32_t parentIdx = ReadOr<int32_t>(parentIndices + static_cast<uintptr_t>(ta.index) * 4, -1);
        int guard = 0;
        while (parentIdx >= 0 && parentIdx < 32768 && guard++ < 48) {
            TrsX pTrs{};
            if (!Read(trsArray + static_cast<uintptr_t>(parentIdx) * 0x30, pTrs)) break;

            // 1. Scale by parent local scale
            float vx = result.x * pTrs.sx;
            float vy = result.y * pTrs.sy;
            float vz = result.z * pTrs.sz;

            // 2. Rotate by parent local quaternion (exact Unity SIMD formula)
            float qx = pTrs.rx, qy = pTrs.ry, qz = pTrs.rz, qw = pTrs.rw;
            float tx = 2.0f * (qy * vz - qz * vy);
            float ty = 2.0f * (qz * vx - qx * vz);
            float tz = 2.0f * (qx * vy - qy * vx);

            result.x = pTrs.px + vx + qw * tx + (qy * tz - qz * ty);
            result.y = pTrs.py + vy + qw * ty + (qz * tx - qx * tz);
            result.z = pTrs.pz + vz + qw * tz + (qx * ty - qy * tx);

            parentIdx = ReadOr<int32_t>(parentIndices + static_cast<uintptr_t>(parentIdx) * 4, -1);
        }
        if (!result.IsFinite()) return false;
        outPos = result;
        return true;
    }

    bool SolveManagedTransformWorldPos(uintptr_t managedTransform, Vec3& outPos) {
        if (!IsValidPtr(managedTransform)) return false;
        uintptr_t cachedPtr = ReadOr<uintptr_t>(managedTransform + 0x10, 0);
        return SolveNativeTransformWorldPos(cachedPtr, outPos);
    }

    bool SolveManagedComponentWorldPos(uintptr_t managedComp, Vec3& outPos) {
        if (!IsValidPtr(managedComp)) return false;
        uintptr_t nativeComp = ReadOr<uintptr_t>(managedComp + 0x10, 0);
        if (!IsValidPtr(nativeComp)) return false;
        uintptr_t nativeGo = ReadOr<uintptr_t>(nativeComp + 0x20, 0);
        if (!IsValidPtr(nativeGo)) return false;
        uintptr_t compArr = ReadOr<uintptr_t>(nativeGo + 0x20, 0);
        if (!IsValidPtr(compArr)) return false;
        uintptr_t nativeTrans = ReadOr<uintptr_t>(compArr + 0x08, 0);
        return SolveNativeTransformWorldPos(nativeTrans, outPos);
    }

    static uint32_t HashUnityTag(uint32_t tag) {
        // Exact Wang integer hash from UnityPlayer.dll FindTaggedObjects (0x1800551C0)
        uint32_t a  = tag * 0x1001U + 0x7ED55D16U;
        uint32_t r8 = (a >> 19) ^ a ^ 0xC761C23CU;
        uint32_t r9 = r8 * 0x21U + 0x165667B1U;
        a = (r9 + 0xD3A2646CU) ^ (r9 << 9);
        a = a * 9U + 0xFD7046C5U;
        return (a >> 16) ^ a ^ 0xB55A4F09U;
    }

    struct BoltStorageView {
        uintptr_t valuesArr  = 0;
        int32_t   arrLen     = 0;
        int32_t   offStorage = -1;
        bool      valid      = false;
    };

    BoltStorageView OpenBoltStorage(uintptr_t networkObj) {
        BoltStorageView v{};
        if (!IsValidPtr(networkObj)) return v;
        uintptr_t rootState = ReadOr<uintptr_t>(networkObj + 0x18, 0);
        if (!IsValidPtr(rootState)) rootState = networkObj;
        uintptr_t frames    = ReadOr<uintptr_t>(rootState + 0x68, 0);
        uintptr_t storage   = IsValidPtr(frames) ? ReadOr<uintptr_t>(frames + 0x10, 0) : 0;
        uintptr_t valuesArr = IsValidPtr(storage) ? ReadOr<uintptr_t>(storage + 0xA0, 0) : 0;
        if (!IsValidPtr(valuesArr)) return v;

        int32_t len = ReadOr<int32_t>(valuesArr + 0x18, 0);
        int32_t off = ReadOr<int32_t>(networkObj + 0x34, -1);
        if (len <= 0 || len > 4096 || off < 0 || off >= len) return v;

        v.valuesArr  = valuesArr;
        v.arrLen     = len;
        v.offStorage = off;
        v.valid      = true;
        return v;
    }

    bool ReadBoltInt(const BoltStorageView& sv, int32_t relPropOffset, int32_t& outVal) {
        if (!sv.valid) return false;
        int32_t idx = sv.offStorage + relPropOffset;
        if (idx < 0 || idx >= sv.arrLen) return false;
        return Read<int32_t>(sv.valuesArr + 0x20 + static_cast<uintptr_t>(idx) * 24, outVal);
    }

    bool ReadBoltByte(const BoltStorageView& sv, int32_t relPropOffset, uint8_t& outVal) {
        if (!sv.valid) return false;
        int32_t idx = sv.offStorage + relPropOffset;
        if (idx < 0 || idx >= sv.arrLen) return false;
        return Read<uint8_t>(sv.valuesArr + 0x20 + static_cast<uintptr_t>(idx) * 24, outVal);
    }

    uintptr_t ReadBoltObjRef(const BoltStorageView& sv, int32_t relPropOffset) {
        if (!sv.valid) return 0;
        int32_t idx = sv.offStorage + relPropOffset;
        if (idx < 0 || idx >= sv.arrLen) return 0;
        return ReadOr<uintptr_t>(sv.valuesArr + 0x20 + static_cast<uintptr_t>(idx) * 24 + 0x10, 0);
    }

    bool ReadMainCamera(WorldSnapshot& snap) {
        uint32_t camTypeBase  = ReadOr<uint32_t>(unityPlayerBase_ + kUP_CameraTypeBase, 0);
        uint32_t camTypeCount = ReadOr<uint32_t>(unityPlayerBase_ + kUP_CameraTypeCount, 1);

        auto tryCameraPtr = [&](uintptr_t camNative, uintptr_t goNative, bool requireEnabled) -> bool {
            if (!IsValidPtr(camNative)) return false;
            if (requireEnabled && ReadOr<uint8_t>(camNative + 0x29, 0) == 0) return false;

            Mat4 view{}, proj{}, cull{};
            if (!Read(camNative + 0x70, view)) return false;
            bool hasProj = Read(camNative + 0xB0, proj) &&
                           std::isfinite(proj.m[5]) && std::fabs(proj.m[5]) >= 0.15f && std::fabs(proj.m[5]) <= 20.0f;
            bool hasCull = Read(camNative + 0x2FC, cull) &&
                           std::isfinite(cull.m[5]) && (std::fabs(cull.m[0]) + std::fabs(cull.m[5])) >= 0.15f;
            if (!hasProj && !hasCull) return false;

            snap.vpMatrix = hasProj ? Mat4::Mul(proj, view) : cull;
            snap.camRight = Vec3(view.m[0], view.m[4], view.m[8]).Normalized();
            snap.camUp    = Vec3(view.m[1], view.m[5], view.m[9]).Normalized();
            snap.camFwd   = Vec3(-view.m[2], -view.m[6], -view.m[10]).Normalized();
            if (hasProj) {
                snap.camFovDeg = 2.0f * std::atan(1.0f / std::fabs(proj.m[5])) * (180.0f / std::numbers::pi_v<float>);
            } else {
                float yScale = Vec3(cull.m[1], cull.m[5], cull.m[9]).Length();
                if (yScale > 0.15f && yScale < 20.0f) {
                    snap.camFovDeg = 2.0f * std::atan(1.0f / yScale) * (180.0f / std::numbers::pi_v<float>);
                }
            }

            Vec3 cpos{};
            bool gotPos = false;
            if (IsValidPtr(goNative)) {
                uintptr_t compArr = ReadOr<uintptr_t>(goNative + 0x20, 0);
                uintptr_t trans   = IsValidPtr(compArr) ? ReadOr<uintptr_t>(compArr + 0x08, 0) : 0;
                gotPos = SolveNativeTransformWorldPos(trans, cpos);
            }
            if (!gotPos) {
                uintptr_t ownerGo = ReadOr<uintptr_t>(camNative + 0x20, 0);
                uintptr_t compArr = IsValidPtr(ownerGo) ? ReadOr<uintptr_t>(ownerGo + 0x20, 0) : 0;
                uintptr_t trans   = IsValidPtr(compArr) ? ReadOr<uintptr_t>(compArr + 0x08, 0) : 0;
                gotPos = SolveNativeTransformWorldPos(trans, cpos);
            }
            if (!gotPos) {
                cpos.x = -(view.m[0] * view.m[12] + view.m[1] * view.m[13] + view.m[2]  * view.m[14]);
                cpos.y = -(view.m[4] * view.m[12] + view.m[5] * view.m[13] + view.m[6]  * view.m[14]);
                cpos.z = -(view.m[8] * view.m[12] + view.m[9] * view.m[13] + view.m[10] * view.m[14]);
            }
            snap.camPos = cpos;
            return snap.camPos.IsFinite() && snap.camFwd.Length() > 0.5f;
        };

        // 1. Primary path: PlayerRoot.get_MainCamera (0x18369FEB0):
        //    SpectatorPlayer (+0x10) or MyPlayer (+0x08) -> _cameraController (+0x48) -> _renderCamera (+0x210) -> m_CachedPtr (+0x10)
        uintptr_t prKlass = ReadOr<uintptr_t>(gameAssemblyBase_ + kGA_PlayerRoot_TypeInfo, 0);
        if (IsValidPtr(prKlass) && (prKlass & 1) == 0 && prKlass > 0x100000000ULL) {
            uintptr_t statics = ReadOr<uintptr_t>(prKlass + kIl2CppClass_StaticFields, 0);
            if (IsValidPtr(statics)) {
                const uintptr_t playerCands[2] = {
                    ReadOr<uintptr_t>(statics + 0x10, 0), // SpectatorPlayer
                    ReadOr<uintptr_t>(statics + 0x08, 0)  // MyPlayer
                };
                for (uintptr_t plr : playerCands) {
                    if (!IsValidPtr(plr)) continue;
                    uintptr_t camCtrl    = ReadOr<uintptr_t>(plr + off_.pr_camController, 0);
                    uintptr_t managedCam = IsValidPtr(camCtrl) ? ReadOr<uintptr_t>(camCtrl + off_.cc_renderCamera, 0) : 0;
                    uintptr_t nativeCam  = IsValidPtr(managedCam) ? ReadOr<uintptr_t>(managedCam + 0x10, 0) : 0;
                    if (tryCameraPtr(nativeCam, 0, false)) return true;
                }
            }
        }

        // 2. Secondary path: UnityPlayer.dll FindMainCamera (0x180592900) via Wang-hashed GameObjectManager tag=5
        uintptr_t gom = ReadOr<uintptr_t>(unityPlayerBase_ + kUP_GameObjectManager, 0);
        if (IsValidPtr(gom)) {
            uintptr_t buckets = ReadOr<uintptr_t>(gom + 0x00, 0);
            uint32_t  mask    = ReadOr<uint32_t>(gom + 0x08, 0);
            if (IsValidPtr(buckets) && mask > 0 && mask < 0x40000) {
                uint32_t hval = HashUnityTag(5U);
                uint32_t edx  = hval & mask;
                uint32_t edi  = hval & 0xFFFFFFFCU;
                for (uint32_t step = 0; step < 64; ++step) {
                    uintptr_t entry = buckets + static_cast<uintptr_t>(edx) * 3ULL;
                    uint32_t  k0    = ReadOr<uint32_t>(entry + 0x00, 0xFFFFFFFFU);
                    uint32_t  t0    = ReadOr<uint32_t>(entry + 0x08, 0xFFFFFFFFU);
                    uintptr_t head  = ReadOr<uintptr_t>(entry + 0x10, 0);
                    if (k0 == edi && t0 == 5U && IsValidPtr(head)) {
                        uintptr_t node = ReadOr<uintptr_t>(head + 0x08, 0);
                        for (int n = 0; n < 64 && IsValidPtr(node) && node != head; ++n) {
                            uintptr_t go = ReadOr<uintptr_t>(node + 0x10, 0);
                            if (IsValidPtr(go) && ReadOr<uint8_t>(go + 0x46, 0) != 0) {
                                uintptr_t compArr = ReadOr<uintptr_t>(go + 0x20, 0);
                                uint64_t  compCnt = ReadOr<uint64_t>(go + 0x30, 0);
                                if (IsValidPtr(compArr) && compCnt >= 2 && compCnt < 32) {
                                    for (uint64_t ci = 1; ci < compCnt; ++ci) {
                                        uint32_t tid = ReadOr<uint32_t>(compArr + ci * 16, 0);
                                        if (camTypeCount == 0 || (tid - camTypeBase) < camTypeCount) {
                                            uintptr_t comp = ReadOr<uintptr_t>(compArr + ci * 16 + 8, 0);
                                            if (tryCameraPtr(comp, go, true)) return true;
                                        }
                                    }
                                }
                            }
                            node = ReadOr<uintptr_t>(node + 0x08, 0);
                        }
                        break;
                    }
                    if (k0 == 0xFFFFFFFFU) break;
                    edx = (edx + 8U) & mask;
                }
            }
        }
        return false;
    }

    std::string ExtractPlayerName(uintptr_t playerRoot, bool isBot, int index, int32_t& outNetTeam) {
        // Primary: _cachedPlayerData (+0x150, Photon.Bolt.NetPlayerData) -> Bolt Storage
        //   +0 = PlayerConnectData (+0x18 = Il2CppString* NickName), +8 = TeamId
        uintptr_t netPlayerData = ReadOr<uintptr_t>(playerRoot + off_.pr_cachedPlayerData, 0);
        if (IsValidPtr(netPlayerData)) {
            BoltStorageView nsv = OpenBoltStorage(netPlayerData);
            if (nsv.valid) {
                int32_t tId = -1;
                if (ReadBoltInt(nsv, 8, tId) && tId >= 0 && tId <= 16) {
                    outNetTeam = tId;
                }
                uintptr_t connectData = ReadBoltObjRef(nsv, 0);
                if (IsValidPtr(connectData)) {
                    std::string nick = ReadIl2CppString(ReadOr<uintptr_t>(connectData + 0x18, 0), 32);
                    if (!nick.empty()) return nick;
                }
            }
        }
        char fallback[32]{};
        std::snprintf(fallback, sizeof(fallback), isBot ? "Bot #%d" : "Operator #%d", index + 1);
        return fallback;
    }

    std::string ExtractWeaponName(uintptr_t armingPtr) {
        if (!IsValidPtr(armingPtr)) return "Primary";
        uintptr_t activeWep = ReadOr<uintptr_t>(armingPtr + off_.pa_activeWeapon, 0);
        if (!IsValidPtr(activeWep)) activeWep = ReadOr<uintptr_t>(armingPtr + off_.pa_mainSlotWeapon, 0);
        if (!IsValidPtr(activeWep)) activeWep = ReadOr<uintptr_t>(armingPtr + off_.pa_primaryWeapon, 0);
        if (!IsValidPtr(activeWep)) return "Unarmed";

        // Read Unity GameObject m_Name from native Component -> GameObject (+0x20) -> m_Name (+0x50 in Unity 2021/2022)
        uintptr_t nativeComp = ReadOr<uintptr_t>(activeWep + 0x10, 0);
        uintptr_t nativeGo   = IsValidPtr(nativeComp) ? ReadOr<uintptr_t>(nativeComp + 0x20, 0) : 0;
        if (IsValidPtr(nativeGo)) {
            uintptr_t namePtr = ReadOr<uintptr_t>(nativeGo + 0x50, 0);
            std::string raw = ReadAscii(namePtr, 48);
            if (!raw.empty()) {
                auto clonePos = raw.find("(Clone)");
                if (clonePos != std::string::npos) raw.erase(clonePos);
                if (raw.rfind("Weapon_", 0) == 0) raw.erase(0, 7);
                if (raw.rfind("Wpn_", 0) == 0)    raw.erase(0, 4);
                if (raw.rfind("View_", 0) == 0)   raw.erase(0, 5);
                for (char& ch : raw) { if (ch == '_') ch = ' '; }
                while (!raw.empty() && raw.back() == ' ') raw.pop_back();
                if (!raw.empty()) return raw;
            }
        }
        return "Primary";
    }

    void ReadMobBones(uintptr_t mobView, Vec3& ioFeet, bool crouched, BoneSkeleton3D& outBones) {
        float fullH = crouched ? 1.18f : 1.68f;
        outBones.head   = ioFeet + Vec3(0.0f, fullH * 0.91f, 0.0f);
        outBones.neck   = ioFeet + Vec3(0.0f, fullH * 0.81f, 0.0f);
        outBones.chest  = ioFeet + Vec3(0.0f, fullH * 0.68f, 0.0f);
        outBones.pelvis = ioFeet + Vec3(0.0f, fullH * 0.50f, 0.0f);
        outBones.lKnee  = ioFeet + Vec3(-0.14f, fullH * 0.26f, 0.0f);
        outBones.rKnee  = ioFeet + Vec3( 0.14f, fullH * 0.26f, 0.0f);
        outBones.lFoot  = ioFeet + Vec3(-0.14f, 0.04f, 0.0f);
        outBones.rFoot  = ioFeet + Vec3( 0.14f, 0.04f, 0.0f);
        outBones.rHand  = outBones.chest + Vec3( 0.22f, -0.10f, 0.12f);
        outBones.lHand  = outBones.chest + Vec3(-0.22f, -0.10f, 0.12f);
        outBones.hasRealBones = false;

        if (!IsValidPtr(mobView)) return;

        // Verified PlayerMobView Transform fields (+0x40..+0x80)
        struct MobPtrBlock {
            uintptr_t head;      // +0x40
            uintptr_t neck;      // +0x48
            uintptr_t gasMask;   // +0x50
            uintptr_t chest;     // +0x58
            uintptr_t spine;     // +0x60
            uintptr_t lKnee;     // +0x68
            uintptr_t rKnee;     // +0x70
            uintptr_t weapR;     // +0x78
            uintptr_t weapL;     // +0x80
        } mb{};

        if (!Read(mobView + 0x40, mb)) return;

        Vec3 pSpine{}, pHead{}, pNeck{}, pChest{}, pLKnee{}, pRKnee{}, pWR{}, pWL{};
        bool okSpine = SolveManagedTransformWorldPos(mb.spine, pSpine);
        bool okHead  = SolveManagedTransformWorldPos(mb.head,  pHead);

        // If root GameObject stayed near world origin while skeleton rig moved, anchor ioFeet to live rig
        if ((ioFeet.Length() < 0.25f || (okSpine && pSpine.Dist(ioFeet) > 4.0f)) && okSpine && pSpine.Length() > 0.25f) {
            ioFeet = Vec3(pSpine.x, pSpine.y - fullH * 0.52f, pSpine.z);
            outBones.head   = ioFeet + Vec3(0.0f, fullH * 0.91f, 0.0f);
            outBones.neck   = ioFeet + Vec3(0.0f, fullH * 0.81f, 0.0f);
            outBones.chest  = ioFeet + Vec3(0.0f, fullH * 0.68f, 0.0f);
            outBones.pelvis = ioFeet + Vec3(0.0f, fullH * 0.50f, 0.0f);
            outBones.lKnee  = ioFeet + Vec3(-0.14f, fullH * 0.26f, 0.0f);
            outBones.rKnee  = ioFeet + Vec3( 0.14f, fullH * 0.26f, 0.0f);
            outBones.lFoot  = ioFeet + Vec3(-0.14f, 0.04f, 0.0f);
            outBones.rFoot  = ioFeet + Vec3( 0.14f, 0.04f, 0.0f);
        }

        int solvedCount = 0;
        if (okHead  && pHead.Dist(ioFeet)  < 3.8f) { outBones.head   = pHead;  ++solvedCount; }
        if (SolveManagedTransformWorldPos(mb.neck,  pNeck)  && pNeck.Dist(ioFeet)  < 3.8f) { outBones.neck  = pNeck;  ++solvedCount; }
        if (SolveManagedTransformWorldPos(mb.chest, pChest) && pChest.Dist(ioFeet) < 3.8f) { outBones.chest = pChest; ++solvedCount; }
        if (okSpine && pSpine.Dist(ioFeet) < 3.8f) { outBones.pelvis = pSpine; ++solvedCount; }
        if (SolveManagedTransformWorldPos(mb.lKnee, pLKnee) && pLKnee.Dist(ioFeet) < 3.8f) {
            outBones.lKnee = pLKnee;
            outBones.lFoot = Vec3(pLKnee.x, ioFeet.y + 0.04f, pLKnee.z);
            ++solvedCount;
        }
        if (SolveManagedTransformWorldPos(mb.rKnee, pRKnee) && pRKnee.Dist(ioFeet) < 3.8f) {
            outBones.rKnee = pRKnee;
            outBones.rFoot = Vec3(pRKnee.x, ioFeet.y + 0.04f, pRKnee.z);
            ++solvedCount;
        }
        if (SolveManagedTransformWorldPos(mb.weapR, pWR) && pWR.Dist(ioFeet) < 3.8f) { outBones.rHand = pWR; }
        if (SolveManagedTransformWorldPos(mb.weapL, pWL) && pWL.Dist(ioFeet) < 3.8f) { outBones.lHand = pWL; }

        outBones.hasRealBones = (solvedCount >= 2);
    }

    int32_t ReadPlayerTeam(uintptr_t pr, const BoltStorageView& stateSv) {
        int32_t teamId = -1;
        if (ReadBoltInt(stateSv, 1, teamId) && teamId >= 0 && teamId <= 16) {
            return teamId;
        }
        uintptr_t cpd = ReadOr<uintptr_t>(pr + off_.pr_cachedPlayerData, 0);
        if (IsValidPtr(cpd)) {
            BoltStorageView nsv = OpenBoltStorage(cpd);
            if (ReadBoltInt(nsv, 8, teamId) && teamId >= 0 && teamId <= 16) {
                return teamId;
            }
        }
        return -1;
    }

    bool ResolvePlayerFeetPos(uintptr_t pr, Vec3& outFeet) {
        if (SolveManagedComponentWorldPos(pr, outFeet) && outFeet.Length() > 0.05f) {
            return true;
        }
        uintptr_t tpPivot = ReadOr<uintptr_t>(pr + off_.pr_thirdPivot, 0);
        if (SolveManagedTransformWorldPos(tpPivot, outFeet) && outFeet.Length() > 0.05f) {
            return true;
        }
        uintptr_t fpPivot = ReadOr<uintptr_t>(pr + off_.pr_firstPivot, 0);
        if (SolveManagedTransformWorldPos(fpPivot, outFeet) && outFeet.Length() > 0.05f) {
            outFeet.y -= 1.45f;
            return true;
        }
        return SolveManagedComponentWorldPos(pr, outFeet);
    }

    void ScanWorld(double nowSec) {
        WorldSnapshot next{};
        next.attached  = true;
        next.pid       = pid_;
        next.gameHwnd  = hwnd_;
        next.rpmPerSec = rpmLastSec_;

        uintptr_t cursorMgr = ReadOr<uintptr_t>(unityPlayerBase_ + kUP_CursorManager, 0);
        if (IsValidPtr(cursorMgr)) {
            next.cursorLocked = (ReadOr<int32_t>(cursorMgr + 0x20, 0) == 1);
        }
        if (!next.cursorLocked && hwnd_ && GetForegroundWindow() == hwnd_) {
            CURSORINFO ci{};
            ci.cbSize = sizeof(ci);
            if (GetCursorInfo(&ci) && (ci.flags & CURSOR_SHOWING) == 0) {
                next.cursorLocked = true;
            }
        }

        RefreshIl2CppReflection(nowSec);
        bool hasCam = ReadMainCamera(next);

        uintptr_t prKlass = ReadOr<uintptr_t>(gameAssemblyBase_ + kGA_PlayerRoot_TypeInfo, 0);
        uintptr_t statics = (IsValidPtr(prKlass) && (prKlass & 1) == 0 && prKlass > 0x100000000ULL)
                            ? ReadOr<uintptr_t>(prKlass + kIl2CppClass_StaticFields, 0) : 0;

        if (IsValidPtr(statics)) {
            // Verified PlayerRoot statics: +0x08 = MyPlayer, +0x10 = SpectatorPlayer, +0x18 = AllPlayers
            uintptr_t myPlayer   = ReadOr<uintptr_t>(statics + 0x08, 0);
            uintptr_t specPlayer = ReadOr<uintptr_t>(statics + 0x10, 0);
            uintptr_t localRef   = IsValidPtr(myPlayer) ? myPlayer : specPlayer;
            uintptr_t allListPtr = ReadOr<uintptr_t>(statics + 0x18, 0);

            // Verify localRef still has a live native Unity MonoBehaviour pointer
            if (IsValidPtr(localRef) && !IsValidPtr(ReadOr<uintptr_t>(localRef + 0x10, 0))) {
                localRef = 0;
            }
            next.localPlayerPtr = localRef;

            if (IsValidPtr(localRef)) {
                ResolvePlayerFeetPos(localRef, next.localPos);
                BoltStorageView localSv = OpenBoltStorage(ReadOr<uintptr_t>(localRef + off_.pr_state, 0));
                next.localTeam = ReadPlayerTeam(localRef, localSv);
            }

            // Collect candidate PlayerRoot pointers from PlayerRoot.AllPlayers (+0x18), with BaseGameModeSystem.Ref.AllPlayers fallback
            std::vector<uintptr_t> ptrs;
            if (IsValidPtr(allListPtr)) {
                uintptr_t itemsArr = ReadOr<uintptr_t>(allListPtr + 0x10, 0);
                int32_t   listSize = ReadOr<int32_t>(allListPtr + 0x18, 0);
                if (IsValidPtr(itemsArr) && listSize > 0 && listSize <= 64) {
                    ptrs.resize(static_cast<size_t>(listSize), 0);
                    if (!ReadRaw(itemsArr + 0x20, ptrs.data(), ptrs.size() * sizeof(uintptr_t))) {
                        ptrs.clear();
                    }
                }
            }
            if (ptrs.empty() && IsValidPtr(off_.baseGmKlass)) {
                uintptr_t gmStatics = ReadOr<uintptr_t>(off_.baseGmKlass + kIl2CppClass_StaticFields, 0);
                uintptr_t gmRef     = IsValidPtr(gmStatics) ? ReadOr<uintptr_t>(gmStatics + 0x00, 0) : 0;
                uintptr_t pdList    = IsValidPtr(gmRef) ? ReadOr<uintptr_t>(gmRef + 0x60, 0) : 0;
                if (IsValidPtr(pdList)) {
                    uintptr_t pdItems = ReadOr<uintptr_t>(pdList + 0x10, 0);
                    int32_t   pdSize  = ReadOr<int32_t>(pdList + 0x18, 0);
                    if (IsValidPtr(pdItems) && pdSize > 0 && pdSize <= 64) {
                        std::vector<uintptr_t> pds(static_cast<size_t>(pdSize), 0);
                        if (ReadRaw(pdItems + 0x20, pds.data(), pds.size() * sizeof(uintptr_t))) {
                            for (uintptr_t pd : pds) {
                                uintptr_t pr = IsValidPtr(pd) ? ReadOr<uintptr_t>(pd + 0x28, 0) : 0;
                                if (IsValidPtr(pr)) ptrs.push_back(pr);
                            }
                        }
                    }
                }
            }

            if (hasCam && (!ptrs.empty() || IsValidPtr(localRef))) {
                next.inMatch = true;
            }

            for (int i = 0; i < static_cast<int>(ptrs.size()); ++i) {
                uintptr_t pr = ptrs[i];
                if (!IsValidPtr(pr)) continue;

                // Ensure native Unity Component & GameObject are alive and active
                uintptr_t nativeComp = ReadOr<uintptr_t>(pr + 0x10, 0);
                if (!IsValidPtr(nativeComp)) continue;
                uintptr_t nativeGo = ReadOr<uintptr_t>(nativeComp + 0x20, 0);
                if (IsValidPtr(nativeGo) && ReadOr<uint8_t>(nativeGo + 0x46, 1) == 0) continue;

                Vec3 feet{};
                if (!ResolvePlayerFeetPos(pr, feet)) continue;

                uintptr_t moveComp = ReadOr<uintptr_t>(pr + off_.pr_movement, 0);
                bool crouched = IsValidPtr(moveComp) && (ReadOr<uint8_t>(moveComp + off_.pm_isCrouch, 0) != 0);

                uintptr_t thirdView = ReadOr<uintptr_t>(pr + off_.pr_thirdView, 0);
                BoneSkeleton3D bones{};
                ReadMobBones(thirdView, feet, crouched, bones);

                // Compute low-pass smoothed velocity over >=35ms windows so 4ms polling ticks never create jitter spikes
                auto& mc = motionCache_[pr];
                if (mc.timestamp <= 0.0) {
                    mc.pos       = feet;
                    mc.timestamp = nowSec;
                } else {
                    double dtVel = nowSec - mc.timestamp;
                    if (dtVel >= 0.035 && dtVel < 0.5) {
                        Vec3 rawVel = (feet - mc.pos) * static_cast<float>(1.0 / dtVel);
                        if (rawVel.Length() < 45.0f) {
                            mc.smoothedVel = mc.smoothedVel * 0.76f + rawVel * 0.24f;
                            if (mc.smoothedVel.Length() < 0.28f) mc.smoothedVel = {};
                        } else {
                            mc.smoothedVel = {};
                        }
                        mc.pos       = feet;
                        mc.timestamp = nowSec;
                    } else if (dtVel >= 0.5) {
                        mc.smoothedVel = {};
                        mc.pos         = feet;
                        mc.timestamp   = nowSec;
                    }
                }

                if (pr == localRef || (!IsValidPtr(localRef) && ReadOr<uint8_t>(pr + off_.pr_hasControl, 0) != 0)) {
                    next.localPlayerPtr = pr;
                    next.localPos       = feet;
                    next.localVel       = mc.smoothedVel;
                    if (next.localTeam < 0) {
                        BoltStorageView lsv = OpenBoltStorage(ReadOr<uintptr_t>(pr + off_.pr_state, 0));
                        next.localTeam = ReadPlayerTeam(pr, lsv);
                    }
                    continue;
                }

                // Exact Combat Master alive & stats read via Bolt PlayerState (0x183670EB0: IsDead == (Health <= 0))
                uintptr_t stateObj = ReadOr<uintptr_t>(pr + off_.pr_state, 0);
                BoltStorageView stateSv = OpenBoltStorage(stateObj);

                float hp       = 100.0f;
                float armor    = 0.0f;
                bool  isDowned = false;
                int32_t netHp  = 0;
                if (ReadBoltInt(stateSv, 0, netHp)) {
                    if (netHp <= 0) continue; // Verified dead operator in Bolt network state
                    hp = static_cast<float>(std::clamp(netHp, 1, 500));
                    int32_t netAr = 0;
                    if (ReadBoltInt(stateSv, 0x3A, netAr) && netAr >= 0 && netAr <= 300) {
                        armor = static_cast<float>(netAr);
                    }
                    uint8_t netDn = 0;
                    if (ReadBoltByte(stateSv, 0x3B, netDn)) {
                        isDowned = (netDn != 0);
                    }
                    uint8_t netCr = 0;
                    if (!crouched && ReadBoltByte(stateSv, 0x11, netCr) && netCr <= 1) {
                        // Optional crouch sync kept via PlayerMovement._isCrouch (+0x90)
                    }
                } else {
                    // Fallback for offline/non-Bolt targets via PlayerHealth cached fields (+0xD0 / +0xCC)
                    uintptr_t healthComp = ReadOr<uintptr_t>(pr + off_.pr_health, 0);
                    if (IsValidPtr(healthComp)) {
                        int32_t rawHp = ReadOr<int32_t>(healthComp + off_.ph_lastHealth, 100);
                        if (rawHp <= 0) continue;
                        if (rawHp <= 500) hp = static_cast<float>(rawHp);
                        int32_t rawAr = ReadOr<int32_t>(healthComp + off_.ph_lastArmor, 0);
                        if (rawAr >= 0 && rawAr <= 300) armor = static_cast<float>(rawAr);
                    }
                }

                if (!bones.hasRealBones && feet.Length() < 0.25f) continue;

                OperatorEntity ent{};
                ent.address    = pr;
                ent.feetPos    = feet;
                ent.bones      = bones;
                ent.headPos    = bones.head;
                ent.velocity   = mc.smoothedVel;
                ent.isCrouched = crouched;
                ent.isDowned   = isDowned;
                ent.health     = std::clamp(hp, 1.0f, 300.0f);
                ent.maxHealth  = (ent.health > 100.0f) ? ent.health : 100.0f;
                ent.armor      = std::clamp(armor, 0.0f, 200.0f);
                ent.maxArmor   = 100.0f;
                ent.isBot      = (ReadOr<uint8_t>(pr + off_.pr_isRealPlayer, 1) == 0);

                int32_t entTeam = ReadPlayerTeam(pr, stateSv);
                int32_t cpdTeam = -1;
                ent.name = ExtractPlayerName(pr, ent.isBot, i, cpdTeam);
                if (entTeam < 0 && cpdTeam >= 0) entTeam = cpdTeam;
                ent.teamId = entTeam;

                if (next.localTeam > 0 && entTeam > 0 && next.localTeam == entTeam) {
                    ent.isTeammate = true;
                }

                Vec3 refOrigin = IsValidPtr(next.localPlayerPtr) && next.localPos.Length() > 0.1f ? next.localPos : next.camPos;
                ent.distanceM  = refOrigin.Dist(feet);
                ent.weaponName = ExtractWeaponName(ReadOr<uintptr_t>(pr + off_.pr_arming, 0));
                next.players.push_back(std::move(ent));
            }
        }

        // Check planted C4 bomb via SNDGameModeSystem (BaseGameModeSystem.Ref) & ActorPlantedBomb
        if (IsValidPtr(off_.baseGmKlass)) {
            uintptr_t gmStat = ReadOr<uintptr_t>(off_.baseGmKlass + kIl2CppClass_StaticFields, 0);
            uintptr_t gmRef  = IsValidPtr(gmStat) ? ReadOr<uintptr_t>(gmStat + 0x00, 0) : 0;
            if (IsValidPtr(gmRef)) {
                uintptr_t gmKl = ReadOr<uintptr_t>(gmRef, 0);
                if (gmKl == off_.sndSystemKlass || ReadAscii(ReadOr<uintptr_t>(gmKl + kIl2CppClass_Name, 0), 32) == "SNDGameModeSystem") {
                    uintptr_t bombGoManaged = ReadOr<uintptr_t>(gmRef + off_.snd_bombGo, 0);
                    uintptr_t bombGoNative  = IsValidPtr(bombGoManaged) ? ReadOr<uintptr_t>(bombGoManaged + 0x10, 0) : 0;
                    if (IsValidPtr(bombGoNative) && ReadOr<uint8_t>(bombGoNative + 0x46, 0) != 0) {
                        Vec3 bpos = ReadOr<Vec3>(gmRef + off_.snd_bombPos, Vec3{});
                        if (!bpos.IsFinite() || bpos.Length() < 0.1f) {
                            uintptr_t compArr = ReadOr<uintptr_t>(bombGoNative + 0x20, 0);
                            uintptr_t trans   = IsValidPtr(compArr) ? ReadOr<uintptr_t>(compArr + 0x08, 0) : 0;
                            SolveNativeTransformWorldPos(trans, bpos);
                        }
                        // Search components of bombGoNative for MonoBehaviour carrying ActorPlantedBomb._timeLeft (+0x28)
                        float tLeft = 0.0f;
                        uintptr_t compArr = ReadOr<uintptr_t>(bombGoNative + 0x20, 0);
                        uint64_t  compCnt = ReadOr<uint64_t>(bombGoNative + 0x30, 0);
                        if (IsValidPtr(compArr) && compCnt >= 2 && compCnt < 24) {
                            for (uint64_t ci = 1; ci < compCnt; ++ci) {
                                uintptr_t nComp = ReadOr<uintptr_t>(compArr + ci * 16 + 8, 0);
                                uintptr_t mComp = IsValidPtr(nComp) ? ReadOr<uintptr_t>(nComp + 0x18, 0) : 0;
                                if (IsValidPtr(mComp) && ReadOr<uintptr_t>(mComp, 0) == off_.plantedBombKlass) {
                                    tLeft = ReadOr<float>(mComp + off_.bomb_timeLeft, 0.0f);
                                    break;
                                }
                            }
                        }
                        if (bpos.IsFinite() && bpos.Length() > 0.1f) {
                            next.bomb.active   = true;
                            next.bomb.position = bpos;
                            next.bomb.timeLeft = (std::isfinite(tLeft) && tLeft > 0.0f && tLeft <= 60.0f) ? tLeft : 40.0f;
                        }
                    }
                }
            }
        }

        next.seq = ++snapSeq_;
        std::lock_guard<std::mutex> lk(mtx_);
        snapshot_ = std::move(next);
    }

    void WorkerLoop() {
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();

        while (running_.load()) {
            double nowSec = std::chrono::duration<double>(clock::now() - t0).count();
            if (nowSec - lastRpmStamp_ >= 1.0) {
                rpmLastSec_   = rpmCounter_;
                rpmCounter_   = 0;
                lastRpmStamp_ = nowSec;
            }

            if (!proc_.Valid()) {
                if (!TryAttach()) {
                    {
                        std::lock_guard<std::mutex> lk(mtx_);
                        snapshot_ = WorldSnapshot{};
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(400));
                    continue;
                }
            } else {
                DWORD exitCode = 0;
                if (!GetExitCodeProcess(proc_.Get(), &exitCode) || exitCode != STILL_ACTIVE) {
                    proc_.Reset();
                    pid_  = 0;
                    hwnd_ = nullptr;
                    continue;
                }
                if (!hwnd_ || !IsWindow(hwnd_)) {
                    hwnd_ = FindUnityWindowForPid(pid_);
                }
            }

            ScanWorld(nowSec);
            std::this_thread::sleep_for(std::chrono::milliseconds(4)); // ~240 Hz world refresh
        }
    }
};

static CombatMasterScanner g_Scanner;

// ============================================================================
// 6. CLOUDY 1.3 PALETTE, MODULES, TOASTS & PROFILE CONFIG SYSTEM
// ============================================================================
enum class ThemeMode : int {
    Dark   = 0,
    Light  = 1,
    System = 2
};

struct Palette {
    ImVec4 bgWindow;
    ImVec4 bgSidebar;
    ImVec4 bgCard;
    ImVec4 bgCardHover;
    ImVec4 bgWidget;
    ImVec4 bgWidgetHover;
    ImVec4 border;
    ImVec4 borderSubtle;
    ImVec4 textPrimary;
    ImVec4 textSecondary;
    ImVec4 textMuted;
    ImVec4 accent;
    ImVec4 accentSecondary;
    ImVec4 accentGlow;
    bool   isLight = false;
};

static bool IsSystemLightTheme() {
    DWORD val = 0;
    DWORD sz  = sizeof(val);
    if (RegGetValueW(HKEY_CURRENT_USER,
                     L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize",
                     L"AppsUseLightTheme", RRF_RT_REG_DWORD, nullptr, &val, &sz) == ERROR_SUCCESS) {
        return val != 0;
    }
    return false;
}

struct ClientThemeState {
    ThemeMode themeMode      = ThemeMode::Dark;
    ImVec4    accentPrimary  = ImVec4(0.95f, 0.28f, 0.40f, 1.0f); // Relinquish Crimson-Rose
    ImVec4    accentSecond   = ImVec4(0.58f, 0.32f, 0.94f, 1.0f); // Arcane Violet
    int       menuToggleVk   = VK_INSERT;
    bool      rebindMenuKey  = false;
    float     menuScale      = 1.0f;

    Palette BuildPalette() const {
        bool light = (themeMode == ThemeMode::Light) || (themeMode == ThemeMode::System && IsSystemLightTheme());
        Palette p{};
        p.isLight         = light;
        p.accent          = accentPrimary;
        p.accentSecondary = accentSecond;
        p.accentGlow      = ImVec4(accentPrimary.x, accentPrimary.y, accentPrimary.z, 0.24f);

        if (!light) {
            p.bgWindow      = ImVec4(0.063f, 0.063f, 0.072f, 0.99f); // #101012 opaque matte shell
            p.bgSidebar     = ImVec4(0.048f, 0.048f, 0.056f, 0.98f); // #0C0C0E
            p.bgCard        = ImVec4(0.088f, 0.088f, 0.102f, 0.98f); // #16161A
            p.bgCardHover   = ImVec4(0.108f, 0.108f, 0.126f, 0.99f);
            p.bgWidget      = ImVec4(0.125f, 0.125f, 0.148f, 1.00f); // #202026
            p.bgWidgetHover = ImVec4(0.158f, 0.158f, 0.186f, 1.00f);
            p.border        = ImVec4(1.000f, 1.000f, 1.000f, 0.082f);
            p.borderSubtle  = ImVec4(1.000f, 1.000f, 1.000f, 0.046f);
            p.textPrimary   = ImVec4(0.955f, 0.958f, 0.972f, 1.00f);
            p.textSecondary = ImVec4(0.675f, 0.685f, 0.725f, 1.00f);
            p.textMuted     = ImVec4(0.455f, 0.465f, 0.505f, 1.00f);
        } else {
            p.bgWindow      = ImVec4(0.955f, 0.958f, 0.968f, 0.99f);
            p.bgSidebar     = ImVec4(0.922f, 0.926f, 0.942f, 0.98f);
            p.bgCard        = ImVec4(0.988f, 0.990f, 0.996f, 0.99f);
            p.bgCardHover   = ImVec4(1.000f, 1.000f, 1.000f, 1.00f);
            p.bgWidget      = ImVec4(0.900f, 0.906f, 0.926f, 1.00f);
            p.bgWidgetHover = ImVec4(0.860f, 0.868f, 0.896f, 1.00f);
            p.border        = ImVec4(0.080f, 0.085f, 0.110f, 0.10f);
            p.borderSubtle  = ImVec4(0.080f, 0.085f, 0.110f, 0.06f);
            p.textPrimary   = ImVec4(0.095f, 0.100f, 0.125f, 1.00f);
            p.textSecondary = ImVec4(0.340f, 0.355f, 0.405f, 1.00f);
            p.textMuted     = ImVec4(0.530f, 0.545f, 0.595f, 1.00f);
        }
        return p;
    }
};

static ClientThemeState g_Theme;

enum class SettingKind : int {
    Toggle = 0,
    SliderFloat,
    SliderInt,
    Combo,
    Color
};

struct ModuleSetting {
    std::string              id;
    std::string              label;
    SettingKind              kind = SettingKind::Toggle;
    bool                     bVal = false;
    float                    fVal = 0.0f;
    float                    fMin = 0.0f;
    float                    fMax = 1.0f;
    std::string              fmt  = "%.1f";
    int                      iVal = 0;
    int                      iMin = 0;
    int                      iMax = 10;
    std::vector<std::string> items{};
    ImVec4                   cVal = ImVec4(1, 1, 1, 1);

    static ModuleSetting MakeToggle(const char* id, const char* lbl, bool def) {
        ModuleSetting s{}; s.id = id; s.label = lbl; s.kind = SettingKind::Toggle; s.bVal = def; return s;
    }
    static ModuleSetting MakeFloat(const char* id, const char* lbl, float def, float mn, float mx, const char* fmt = "%.1f") {
        ModuleSetting s{}; s.id = id; s.label = lbl; s.kind = SettingKind::SliderFloat;
        s.fVal = def; s.fMin = mn; s.fMax = mx; s.fmt = fmt; return s;
    }
    static ModuleSetting MakeCombo(const char* id, const char* lbl, int def, std::initializer_list<const char*> opts) {
        ModuleSetting s{}; s.id = id; s.label = lbl; s.kind = SettingKind::Combo; s.iVal = def;
        for (const char* o : opts) s.items.emplace_back(o);
        return s;
    }
    static ModuleSetting MakeColor(const char* id, const char* lbl, ImVec4 def) {
        ModuleSetting s{}; s.id = id; s.label = lbl; s.kind = SettingKind::Color; s.cVal = def; return s;
    }
};

enum class BindMode : int {
    Toggle = 0,
    Hold   = 1
};

struct Module {
    std::string                id;
    std::string                name;
    std::string                desc;
    int                        category  = 0; // 0=Combat, 1=Visuals, 2=Misc
    int                        subtab    = 0;
    bool                       enabled   = false;
    int                        keybind   = 0; // VK code, 0 = None
    BindMode                   bindMode  = BindMode::Toggle;
    bool                       waitingKey = false;
    bool                       expanded  = false;
    bool                       keyWasDown = false;
    std::vector<ModuleSetting> settings{};

    ModuleSetting* Find(const char* sid) {
        for (auto& s : settings) { if (s.id == sid) return &s; }
        return nullptr;
    }
    const ModuleSetting* Find(const char* sid) const {
        for (const auto& s : settings) { if (s.id == sid) return &s; }
        return nullptr;
    }
    bool GetBool(const char* sid, bool fb = false) const {
        const auto* s = Find(sid); return s ? s->bVal : fb;
    }
    float GetFloat(const char* sid, float fb = 0.0f) const {
        const auto* s = Find(sid); return s ? s->fVal : fb;
    }
    int GetCombo(const char* sid, int fb = 0) const {
        const auto* s = Find(sid); return s ? s->iVal : fb;
    }
    ImVec4 GetColor(const char* sid, ImVec4 fb = ImVec4(1, 1, 1, 1)) const {
        const auto* s = Find(sid); return s ? s->cVal : fb;
    }
    std::string ModeBadge() const {
        for (const auto& s : settings) {
            if (s.kind == SettingKind::Combo && s.iVal >= 0 && s.iVal < static_cast<int>(s.items.size())) {
                return s.items[s.iVal];
            }
        }
        return {};
    }
};

struct ToastNotification {
    std::string title;
    std::string detail;
    bool        positive = true;
    float       age      = 0.0f;
    float       duration = 2.4f;
    float       slideX   = 1.0f;
};

static std::vector<Module>            g_Modules;
static std::vector<ToastNotification> g_Toasts;

static void PushToast(const std::string& title, const std::string& detail, bool positive = true) {
    ToastNotification t{};
    t.title    = title;
    t.detail   = detail;
    t.positive = positive;
    t.slideX   = 1.0f;
    g_Toasts.push_back(std::move(t));
    if (g_Toasts.size() > 6) g_Toasts.erase(g_Toasts.begin());
}

static Module* FindModule(const char* id) {
    for (auto& m : g_Modules) { if (m.id == id) return &m; }
    return nullptr;
}

static std::string VkToString(int vk) {
    if (vk <= 0) return "None";
    switch (vk) {
        case VK_LBUTTON:    return "M1";
        case VK_RBUTTON:    return "M2";
        case VK_MBUTTON:    return "M3";
        case VK_XBUTTON1:   return "M4";
        case VK_XBUTTON2:   return "M5";
        case VK_INSERT:     return "INS";
        case VK_DELETE:     return "DEL";
        case VK_HOME:       return "HOME";
        case VK_END:        return "END";
        case VK_PRIOR:      return "PGUP";
        case VK_NEXT:       return "PGDN";
        case VK_UP:         return "UP";
        case VK_DOWN:       return "DOWN";
        case VK_LEFT:       return "LEFT";
        case VK_RIGHT:      return "RIGHT";
        case VK_SHIFT:
        case VK_LSHIFT:     return "LSHIFT";
        case VK_RSHIFT:     return "RSHIFT";
        case VK_CONTROL:
        case VK_LCONTROL:   return "LCTRL";
        case VK_RCONTROL:   return "RCTRL";
        case VK_MENU:
        case VK_LMENU:      return "LALT";
        case VK_RMENU:      return "RALT";
        case VK_CAPITAL:    return "CAPS";
        case VK_TAB:        return "TAB";
        case VK_SPACE:      return "SPACE";
        case VK_MULTIPLY:   return "NP *";
        case VK_ADD:        return "NP +";
        case VK_SUBTRACT:   return "NP -";
        case VK_DECIMAL:    return "NP .";
        case VK_DIVIDE:     return "NP /";
        case VK_OEM_1:      return ";";
        case VK_OEM_PLUS:   return "=";
        case VK_OEM_COMMA:  return ",";
        case VK_OEM_MINUS:  return "-";
        case VK_OEM_PERIOD: return ".";
        case VK_OEM_2:      return "/";
        case VK_OEM_3:      return "`";
        case VK_OEM_4:      return "[";
        case VK_OEM_5:      return "\\";
        case VK_OEM_6:      return "]";
        case VK_OEM_7:      return "'";
        default: break;
    }
    if (vk >= 'A' && vk <= 'Z') return std::string(1, static_cast<char>(vk));
    if (vk >= '0' && vk <= '9') return std::string(1, static_cast<char>(vk));
    if (vk >= VK_F1 && vk <= VK_F12) return "F" + std::to_string(vk - VK_F1 + 1);
    if (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD9) return "NP " + std::to_string(vk - VK_NUMPAD0);

    UINT scan = MapVirtualKeyA(static_cast<UINT>(vk), MAPVK_VK_TO_VSC);
    if (scan != 0) {
        LONG lParam = static_cast<LONG>(scan << 16);
        char buf[32]{};
        if (GetKeyNameTextA(lParam, buf, sizeof(buf)) > 0) {
            return buf;
        }
    }
    return "VK" + std::to_string(vk);
}

static bool IsBindableVk(int vk) {
    if (vk == VK_RBUTTON || vk == VK_MBUTTON || vk == VK_XBUTTON1 || vk == VK_XBUTTON2) return true;
    if (vk >= '0' && vk <= '9') return true;
    if (vk >= 'A' && vk <= 'Z') return true;
    if (vk >= VK_F1 && vk <= VK_F12) return true;
    if (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD9) return true;
    switch (vk) {
        case VK_TAB:
        case VK_CAPITAL:
        case VK_SPACE:
        case VK_PRIOR:
        case VK_NEXT:
        case VK_END:
        case VK_HOME:
        case VK_LEFT:
        case VK_UP:
        case VK_RIGHT:
        case VK_DOWN:
        case VK_INSERT:
        case VK_DELETE:
        case VK_LSHIFT:
        case VK_RSHIFT:
        case VK_LCONTROL:
        case VK_RCONTROL:
        case VK_LMENU:
        case VK_RMENU:
        case VK_MULTIPLY:
        case VK_ADD:
        case VK_SUBTRACT:
        case VK_DECIMAL:
        case VK_DIVIDE:
        case VK_OEM_1:
        case VK_OEM_PLUS:
        case VK_OEM_COMMA:
        case VK_OEM_MINUS:
        case VK_OEM_PERIOD:
        case VK_OEM_2:
        case VK_OEM_3:
        case VK_OEM_4:
        case VK_OEM_5:
        case VK_OEM_6:
        case VK_OEM_7:
            return true;
        default:
            return false;
    }
}

static void InitModules() {
    g_Modules.clear();

    // ---------------- CATEGORY 0: COMBAT (Subtabs: 0=Aim, 1=Trigger) ----------------
    {
        Module m{};
        m.id       = "aimbot";
        m.name     = "Precision Lock";
        m.desc     = "Angular camera tracking with bone priority & recoil control";
        m.category = 0;
        m.subtab   = 0;
        m.settings = {
            ModuleSetting::MakeCombo("activation", "Activation", 0, { "Right Click", "Left Click", "L+R Click", "Always", "Hold Bind" }),
            ModuleSetting::MakeCombo("bone",       "Hitbox Bone", 0, { "Head", "Neck", "Chest", "Pelvis", "Nearest" }),
            ModuleSetting::MakeFloat("fov",        "FOV Radius", 10.5f, 1.5f, 60.0f, "%.1f deg"),
            ModuleSetting::MakeToggle("draw_fov",  "Draw FOV Ring", true),
            ModuleSetting::MakeFloat("smooth",     "Smoothness", 8.0f, 1.5f, 30.0f, "%.1f"),
            ModuleSetting::MakeFloat("sens_scale", "Sensitivity Scale", 1.00f, 0.15f, 2.80f, "%.2fx"),
            ModuleSetting::MakeToggle("predict",   "Velocity Prediction", false),
            ModuleSetting::MakeFloat("lead_scale", "Lead Scale", 0.65f, 0.10f, 1.50f, "%.2fx"),
            ModuleSetting::MakeFloat("rcs_pitch",  "Recoil Pitch Assist", 18.0f, 0.0f, 100.0f, "%.0f%%"),
            ModuleSetting::MakeToggle("sticky",    "Sticky Target Lock", true),
            ModuleSetting::MakeToggle("team_chk",  "Enemies Only", true),
            ModuleSetting::MakeFloat("max_dist",   "Max Range", 180.0f, 20.0f, 350.0f, "%.0f m"),
            ModuleSetting::MakeColor("fov_col",    "FOV Ring Colour", ImVec4(0.95f, 0.28f, 0.40f, 0.72f))
        };
        g_Modules.push_back(std::move(m));
    }
    {
        Module m{};
        m.id       = "target_hud";
        m.name     = "Target HUD";
        m.desc     = "Draggable CS2 tactical operator card with HP damage trail & armour bar";
        m.category = 0;
        m.subtab   = 0;
        m.settings = {
            ModuleSetting::MakeCombo("style",       "Card Style", 0, { "Relinquish Glass", "CS2 Compact" }),
            ModuleSetting::MakeToggle("silhouette", "Operator Avatar", true),
            ModuleSetting::MakeToggle("armor_bar",  "Show Armour Bar", true)
        };
        g_Modules.push_back(std::move(m));
    }
    {
        Module m{};
        m.id       = "triggerbot";
        m.name     = "Auto Trigger";
        m.desc     = "3D ray-capsule intersection firing when sightline crosses target bones";
        m.category = 0;
        m.subtab   = 1;
        m.settings = {
            ModuleSetting::MakeCombo("mode",       "Trigger Mode", 0, { "Auto Fire", "Hold Bind" }),
            ModuleSetting::MakeFloat("radius",     "Capsule Tolerance", 0.20f, 0.08f, 0.45f, "%.2f m"),
            ModuleSetting::MakeFloat("delay_ms",   "Reaction Delay", 25.0f, 0.0f, 250.0f, "%.0f ms"),
            ModuleSetting::MakeFloat("cadence_ms", "Shot Interval", 95.0f, 35.0f, 360.0f, "%.0f ms"),
            ModuleSetting::MakeToggle("ads_only",  "Scoped / ADS Only", false),
            ModuleSetting::MakeToggle("team_chk",  "Enemies Only", true)
        };
        g_Modules.push_back(std::move(m));
    }

    // ---------------- CATEGORY 1: VISUALS (Subtabs: 0=Players, 1=World, 2=HUD) ----------------
    {
        Module m{};
        m.id       = "player_esp";
        m.name     = "Operator ESP";
        m.desc     = "CS2 corner/2D/3D bounding boxes, bone rig skeleton, HP/armour & tags";
        m.category = 1;
        m.subtab   = 0;
        m.settings = {
            ModuleSetting::MakeCombo("box_mode",   "Box Style", 0, { "CS2 Corners", "2D Box", "3D Box", "None" }),
            ModuleSetting::MakeToggle("box_fill",  "Subtle Box Fill", true),
            ModuleSetting::MakeToggle("skeleton",  "Bone Skeleton", true),
            ModuleSetting::MakeToggle("head_dot",  "Head Hitbox Dot", true),
            ModuleSetting::MakeToggle("hp_bar",    "Health Bar", true),
            ModuleSetting::MakeToggle("armor_bar", "Armour Bar", true),
            ModuleSetting::MakeToggle("name_tag",  "Operator Name", true),
            ModuleSetting::MakeToggle("weap_tag",  "Active Weapon", true),
            ModuleSetting::MakeToggle("dist_tag",  "Distance Readout", true),
            ModuleSetting::MakeToggle("enemy_only","Enemies Only", true),
            ModuleSetting::MakeToggle("oof_arrows","Offscreen Arrows", true),
            ModuleSetting::MakeFloat("oof_radius", "Arrow Orbit Radius", 185.0f, 85.0f, 380.0f, "%.0f px"),
            ModuleSetting::MakeFloat("max_dist",   "Max Render Range", 250.0f, 30.0f, 400.0f, "%.0f m"),
            ModuleSetting::MakeColor("col_enemy",  "Enemy Colour",    ImVec4(0.96f, 0.29f, 0.41f, 1.0f)),
            ModuleSetting::MakeColor("col_team",   "Teammate Colour", ImVec4(0.28f, 0.78f, 0.96f, 1.0f)),
            ModuleSetting::MakeColor("col_locked", "Locked Colour",   ImVec4(1.00f, 0.82f, 0.28f, 1.0f)),
            ModuleSetting::MakeColor("col_bones",  "Skeleton Colour", ImVec4(0.96f, 0.96f, 0.99f, 0.90f))
        };
        g_Modules.push_back(std::move(m));
    }
    {
        Module m{};
        m.id       = "snaplines";
        m.name     = "Snaplines";
        m.desc     = "Tactical sight vectors linking screen anchor to active operators";
        m.category = 1;
        m.subtab   = 0;
        m.settings = {
            ModuleSetting::MakeCombo("origin",     "Anchor Origin", 0, { "Bottom", "Crosshair", "Top" }),
            ModuleSetting::MakeToggle("bezier",    "Curved Arc", false),
            ModuleSetting::MakeToggle("enemy_only","Enemies Only", true),
            ModuleSetting::MakeColor("col",        "Line Colour", ImVec4(0.95f, 0.28f, 0.40f, 0.68f))
        };
        g_Modules.push_back(std::move(m));
    }
    {
        Module m{};
        m.id       = "bomb_world";
        m.name     = "Bomb & World";
        m.desc     = "Search & Destroy planted C4 world indicator, defuse window & blast bar";
        m.category = 1;
        m.subtab   = 1;
        m.settings = {
            ModuleSetting::MakeToggle("world_pin", "3D World C4 Marker", true),
            ModuleSetting::MakeToggle("top_bar",   "Top Countdown Bar", true),
            ModuleSetting::MakeToggle("dmg_calc",  "Blast Radius Estimate", true)
        };
        g_Modules.push_back(std::move(m));
    }
    {
        Module m{};
        m.id       = "radar_2d";
        m.name     = "2D Radar";
        m.desc     = "Draggable camera-aligned CS2 tactical radar with FOV cone & C4 pin";
        m.category = 1;
        m.subtab   = 2;
        m.settings = {
            ModuleSetting::MakeCombo("shape",      "Radar Frame", 0, { "Rounded Square", "Circular" }),
            ModuleSetting::MakeFloat("size",       "Radar Size", 190.0f, 130.0f, 300.0f, "%.0f px"),
            ModuleSetting::MakeFloat("range",      "Scan Range", 75.0f, 25.0f, 180.0f, "%.0f m"),
            ModuleSetting::MakeToggle("fov_cone",  "Camera FOV Cone", true),
            ModuleSetting::MakeToggle("enemy_only","Enemies Only", false)
        };
        g_Modules.push_back(std::move(m));
    }
    {
        Module m{};
        m.id       = "crosshair";
        m.name     = "Crosshair";
        m.desc     = "Custom CS2 static-dynamic reticle with lock & fire gap feedback";
        m.category = 1;
        m.subtab   = 2;
        m.settings = {
            ModuleSetting::MakeCombo("style",      "Reticle Style", 0, { "CS2 Classic", "Circle + Dot", "Precision Dot", "T-Cross" }),
            ModuleSetting::MakeFloat("length",     "Arm Length", 6.5f, 2.0f, 18.0f, "%.1f px"),
            ModuleSetting::MakeFloat("gap",        "Center Gap", 3.5f, 0.0f, 14.0f, "%.1f px"),
            ModuleSetting::MakeFloat("thick",      "Stroke Width", 1.5f, 1.0f, 4.0f, "%.1f px"),
            ModuleSetting::MakeToggle("dynamic",   "Dynamic Expand", true),
            ModuleSetting::MakeToggle("outline",   "Crisp Outline", true),
            ModuleSetting::MakeColor("col",        "Reticle Colour", ImVec4(0.30f, 0.98f, 0.62f, 0.95f))
        };
        g_Modules.push_back(std::move(m));
    }
    {
        Module m{};
        m.id       = "client_hud";
        m.name     = "Client HUD";
        m.desc     = "Relinquish crest watermark, stepped module arraylist & toggle toasts";
        m.category = 1;
        m.subtab   = 2;
        m.enabled  = true; // Default ON matching mc_edu_overlay
        m.settings = {
            ModuleSetting::MakeToggle("watermark", "Show Watermark", true),
            ModuleSetting::MakeToggle("arraylist", "Show Arraylist", true),
            ModuleSetting::MakeCombo("side",       "Arraylist Anchor", 0, { "Top Right", "Top Left" }),
            ModuleSetting::MakeToggle("toasts",    "Toggle Notifications", true)
        };
        g_Modules.push_back(std::move(m));
    }
    {
        Module m{};
        m.id       = "telemetry_hud";
        m.name     = "Telemetry HUD";
        m.desc     = "Draggable live process status, XYZ coordinates, speed & RPM metrics";
        m.category = 1;
        m.subtab   = 2;
        m.settings = {
            ModuleSetting::MakeToggle("coords",    "Show XYZ & Velocity", true),
            ModuleSetting::MakeToggle("rpm",       "Show Memory Telemetry", true)
        };
        g_Modules.push_back(std::move(m));
    }

    // ---------------- CATEGORY 2: MISC (Subtabs: 0=Utility) ----------------
    {
        Module m{};
        m.id       = "prox_alert";
        m.name     = "Proximity Alert";
        m.desc     = "Warns when an enemy operator breaches your close-quarters perimeter";
        m.category = 2;
        m.subtab   = 0;
        m.settings = {
            ModuleSetting::MakeFloat("radius",     "Warning Radius", 18.0f, 5.0f, 45.0f, "%.0f m"),
            ModuleSetting::MakeToggle("chime",     "Audio Alert Tone", true)
        };
        g_Modules.push_back(std::move(m));
    }
    {
        Module m{};
        m.id       = "free_cursor";
        m.name     = "Free Cursor";
        m.desc     = "Unlocks overlay cursor to reposition HUD widgets without opening menu";
        m.category = 2;
        m.subtab   = 0;
        m.bindMode = BindMode::Hold;
        m.settings = {
            ModuleSetting::MakeToggle("dim_bg",    "Subtle Screen Dim", false)
        };
        g_Modules.push_back(std::move(m));
    }
}

// ============================================================================
// 7. PROFILE CONFIG SERIALIZATION (RELINQUISH_CFG_V1)
// ============================================================================
class ConfigManager {
public:
    fs::path                 configDir;
    std::vector<std::string> profiles;
    int                      selectedProfile = 0;
    char                     newProfileBuf[48] = "legit_cs2";
    bool                     filterCombat  = true;
    bool                     filterVisuals = true;
    bool                     filterMisc    = true;
    bool                     filterTheme   = true;

    void Init(const fs::path& baseDir) {
        configDir = baseDir / "configs";
        std::error_code ec;
        fs::create_directories(configDir, ec);
        RefreshList();
    }

    void RefreshList() {
        profiles.clear();
        std::error_code ec;
        if (fs::exists(configDir, ec)) {
            for (const auto& entry : fs::directory_iterator(configDir, ec)) {
                if (entry.is_regular_file() && entry.path().extension() == ".cfg") {
                    profiles.push_back(entry.path().stem().string());
                }
            }
        }
        std::sort(profiles.begin(), profiles.end());
        if (selectedProfile >= static_cast<int>(profiles.size())) {
            selectedProfile = std::max(0, static_cast<int>(profiles.size()) - 1);
        }
    }

    bool Save(const std::string& name) {
        if (name.empty()) return false;
        std::error_code ec;
        fs::create_directories(configDir, ec);
        fs::path file = configDir / (name + ".cfg");
        std::ofstream out(file, std::ios::trunc);
        if (!out.is_open()) return false;

        out << "RELINQUISH_CFG_V1\n";
        out << "theme.mode=" << static_cast<int>(g_Theme.themeMode) << "\n";
        out << "theme.accent=" << g_Theme.accentPrimary.x << "," << g_Theme.accentPrimary.y << "," << g_Theme.accentPrimary.z << "," << g_Theme.accentPrimary.w << "\n";
        out << "theme.accent2=" << g_Theme.accentSecond.x << "," << g_Theme.accentSecond.y << "," << g_Theme.accentSecond.z << "," << g_Theme.accentSecond.w << "\n";
        out << "theme.menu_vk=" << g_Theme.menuToggleVk << "\n";
        out << "shader.backdrop=" << (g_Shaders.backdropEnabled ? 1 : 0) << "\n";
        out << "shader.atmos=" << (g_Shaders.screenAtmosEnabled ? 1 : 0) << "\n";
        out << "shader.preset=" << g_Shaders.screenAtmosPreset << "\n";
        out << "anim.preset=" << static_cast<int>(g_Anim.preset) << "\n";
        out << "anim.speed=" << g_Anim.speedScale << "\n";
        out << "audio.enabled=" << (g_Sound.enabled ? 1 : 0) << "\n";
        out << "audio.vol=" << g_Sound.volume << "\n";
        out << "audio.pack=" << static_cast<int>(g_Sound.pack) << "\n";

        for (const auto& m : g_Modules) {
            out << "mod." << m.id << ".enabled=" << (m.enabled ? 1 : 0) << "\n";
            out << "mod." << m.id << ".key=" << m.keybind << "\n";
            out << "mod." << m.id << ".bmode=" << static_cast<int>(m.bindMode) << "\n";
            for (const auto& s : m.settings) {
                std::string prefix = "set." + m.id + "." + s.id + "=";
                if (s.kind == SettingKind::Toggle)      out << prefix << (s.bVal ? 1 : 0) << "\n";
                else if (s.kind == SettingKind::SliderFloat) out << prefix << s.fVal << "\n";
                else if (s.kind == SettingKind::SliderInt || s.kind == SettingKind::Combo) out << prefix << s.iVal << "\n";
                else if (s.kind == SettingKind::Color)  out << prefix << s.cVal.x << "," << s.cVal.y << "," << s.cVal.z << "," << s.cVal.w << "\n";
            }
        }
        out.close();
        RefreshList();
        return true;
    }

    bool Load(const std::string& name) {
        fs::path file = configDir / (name + ".cfg");
        std::ifstream in(file);
        if (!in.is_open()) return false;

        std::string header;
        std::getline(in, header);
        if (header.find("RELINQUISH_CFG_V1") == std::string::npos) return false;

        auto parseVec4 = [](const std::string& v, ImVec4& dst) {
            float a = dst.x, b = dst.y, c = dst.z, d = dst.w;
            if (std::sscanf(v.c_str(), "%f,%f,%f,%f", &a, &b, &c, &d) == 4) {
                dst = ImVec4(a, b, c, d);
            }
        };

        std::string line;
        while (std::getline(in, line)) {
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string k = line.substr(0, eq);
            std::string v = line.substr(eq + 1);

            if (filterTheme) {
                if (k == "theme.mode")         g_Theme.themeMode = static_cast<ThemeMode>(std::clamp(std::stoi(v), 0, 2));
                else if (k == "theme.accent")  parseVec4(v, g_Theme.accentPrimary);
                else if (k == "theme.accent2") parseVec4(v, g_Theme.accentSecond);
                else if (k == "theme.menu_vk") g_Theme.menuToggleVk = std::stoi(v);
                else if (k == "shader.backdrop") g_Shaders.backdropEnabled = (std::stoi(v) != 0);
                else if (k == "shader.atmos")    g_Shaders.screenAtmosEnabled = (std::stoi(v) != 0);
                else if (k == "shader.preset")   g_Shaders.screenAtmosPreset = std::clamp(std::stoi(v), 0, 3);
                else if (k == "anim.preset")     g_Anim.preset = static_cast<SpringPreset>(std::clamp(std::stoi(v), 0, 3));
                else if (k == "anim.speed")      g_Anim.speedScale = std::clamp(std::stof(v), 0.35f, 2.5f);
                else if (k == "audio.enabled")   g_Sound.enabled = (std::stoi(v) != 0);
                else if (k == "audio.vol")       g_Sound.volume = std::clamp(std::stof(v), 0.0f, 1.0f);
                else if (k == "audio.pack")      g_Sound.pack = static_cast<SoundPack>(std::clamp(std::stoi(v), 0, 3));
            }

            if (k.rfind("mod.", 0) == 0) {
                auto secondDot = k.find('.', 4);
                if (secondDot == std::string::npos) continue;
                std::string mid  = k.substr(4, secondDot - 4);
                std::string prop = k.substr(secondDot + 1);
                Module* m = FindModule(mid.c_str());
                if (!m) continue;
                if ((m->category == 0 && !filterCombat) ||
                    (m->category == 1 && !filterVisuals) ||
                    (m->category == 2 && !filterMisc)) continue;

                if (prop == "enabled")    m->enabled  = (std::stoi(v) != 0);
                else if (prop == "key")   m->keybind  = std::stoi(v);
                else if (prop == "bmode") m->bindMode = static_cast<BindMode>(std::clamp(std::stoi(v), 0, 1));
            } else if (k.rfind("set.", 0) == 0) {
                auto d1 = k.find('.', 4);
                auto d2 = (d1 != std::string::npos) ? k.find('.', d1 + 1) : std::string::npos;
                if (d2 == std::string::npos) continue;
                std::string mid = k.substr(4, d1 - 4);
                std::string sid = k.substr(d1 + 1);
                Module* m = FindModule(mid.c_str());
                if (!m) continue;
                if ((m->category == 0 && !filterCombat) ||
                    (m->category == 1 && !filterVisuals) ||
                    (m->category == 2 && !filterMisc)) continue;

                ModuleSetting* s = m->Find(sid.c_str());
                if (!s) continue;
                if (s->kind == SettingKind::Toggle) s->bVal = (std::stoi(v) != 0);
                else if (s->kind == SettingKind::SliderFloat) s->fVal = std::clamp(std::stof(v), s->fMin, s->fMax);
                else if (s->kind == SettingKind::SliderInt || s->kind == SettingKind::Combo) s->iVal = std::stoi(v);
                else if (s->kind == SettingKind::Color) parseVec4(v, s->cVal);
            }
        }
        return true;
    }

    bool Delete(const std::string& name) {
        std::error_code ec;
        bool ok = fs::remove(configDir / (name + ".cfg"), ec);
        RefreshList();
        return ok;
    }
};

static ConfigManager g_Configs;

// ============================================================================
// 8. COMBAT ENGINE (PRECISION LOCK, AUTO TRIGGER) & CS2 VISUAL OVERLAY ENGINE
// ============================================================================
struct CombatRuntimeState {
    uintptr_t lockedAddress   = 0;
    bool      hasLockedTarget = false;
    OperatorEntity lockedSnapshot{};
    float     subPixelX       = 0.0f;
    float     subPixelY       = 0.0f;
    uint64_t  lastAimSnapSeq  = 0;
    float     snapSentDegX    = 0.0f;
    float     snapSentDegY    = 0.0f;
    double    firstSightTime  = 0.0;
    double    lastShotTime    = 0.0;
    float     hpGhostTrail    = 100.0f;
    float     hpSmooth        = 100.0f;
    float     armorSmooth     = 0.0f;
    double    lastProxChime   = 0.0;

    // Draggable HUD widget positions
    ImVec2    targetHudPos    = ImVec2(-1.0f, -1.0f);
    ImVec2    radarPos        = ImVec2(24.0f, 64.0f);
    ImVec2    telemetryPos    = ImVec2(24.0f, 280.0f);
};

static CombatRuntimeState g_Combat;

static Vec3 SelectTargetBone(const OperatorEntity& ent, int boneMode, const Mat4& vp, float sw, float sh) {
    switch (boneMode) {
        case 0: return ent.bones.head;
        case 1: return ent.bones.neck;
        case 2: return ent.bones.chest;
        case 3: return ent.bones.pelvis;
        default: {
            const Vec3 candidates[4] = { ent.bones.head, ent.bones.neck, ent.bones.chest, ent.bones.pelvis };
            Vec2 center(sw * 0.5f, sh * 0.5f);
            float bestDist = 1e9f;
            Vec3 best = ent.bones.head;
            for (const Vec3& c : candidates) {
                Vec2 sc{};
                if (vp.WorldToScreen(c, sw, sh, sc)) {
                    float d = (sc - center).Length();
                    if (d < bestDist) { bestDist = d; best = c; }
                }
            }
            return best;
        }
    }
}

static void SendMouseDelta(int dx, int dy) {
    if (dx == 0 && dy == 0) return;
    INPUT inp{};
    inp.type       = INPUT_MOUSE;
    inp.mi.dx      = dx;
    inp.mi.dy      = dy;
    inp.mi.dwFlags = MOUSEEVENTF_MOVE;
    SendInput(1, &inp, sizeof(INPUT));
}

static void TapMouseClick() {
    INPUT inps[2]{};
    inps[0].type       = INPUT_MOUSE;
    inps[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    inps[1].type       = INPUT_MOUSE;
    inps[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(2, inps, sizeof(INPUT));
}

static void RunCombatAndVisuals(ImDrawList* bgList, const WorldSnapshot& world, const Palette& pal,
                                float sw, float sh, float dt, double nowSec, bool menuOpen, bool freeCursorActive) {
    Module* mAim     = FindModule("aimbot");
    Module* mTrig    = FindModule("triggerbot");
    Module* mThud    = FindModule("target_hud");
    Module* mEsp     = FindModule("player_esp");
    Module* mSnap    = FindModule("snaplines");
    Module* mBomb    = FindModule("bomb_world");
    Module* mRadar   = FindModule("radar_2d");
    Module* mCross   = FindModule("crosshair");
    Module* mTele    = FindModule("telemetry_hud");
    Module* mProx    = FindModule("prox_alert");

    ImVec2 screenCenter(sw * 0.5f, sh * 0.5f);
    float degToPx = (sh * 0.5f) / std::tan(std::clamp(world.camFovDeg, 30.0f, 130.0f) * 0.5f * (std::numbers::pi_v<float> / 180.0f));

    // ---------------- 1. TARGET ACQUISITION & PRECISION LOCK ----------------
    bool aimModuleOn = mAim && mAim->enabled;
    bool aimEnabled  = aimModuleOn && world.inMatch;
    float aimFovDeg  = mAim ? mAim->GetFloat("fov", 10.5f) : 10.0f;
    float aimFovPx   = std::tan(aimFovDeg * (std::numbers::pi_v<float> / 180.0f)) * degToPx;

    if (aimModuleOn && mAim->GetBool("draw_fov", true)) {
        float animR = g_Anim.StepKey("hud.fov_radius", aimFovPx, dt);
        ImVec4 fc = mAim->GetColor("fov_col", pal.accent);
        bgList->AddCircle(screenCenter, animR + 1.0f, IM_COL32(0, 0, 0, static_cast<int>(fc.w * 110.0f)), 96, 2.2f);
        bgList->AddCircle(screenCenter, animR, ImGui::ColorConvertFloat4ToU32(fc), 96, 1.4f);
    }

    const OperatorEntity* bestTarget = nullptr;
    Vec2 bestTargetScreen{};
    Vec3 bestAimWorld{};

    if (world.inMatch && !world.players.empty()) {
        int   boneMode = mAim ? mAim->GetCombo("bone", 0) : 0;
        bool  teamChk  = mAim ? mAim->GetBool("team_chk", true) : true;
        float maxRange = mAim ? mAim->GetFloat("max_dist", 180.0f) : 180.0f;
        bool  predict  = mAim ? mAim->GetBool("predict", false) : false;
        float leadMul  = mAim ? mAim->GetFloat("lead_scale", 0.65f) : 0.65f;
        bool  sticky   = mAim ? mAim->GetBool("sticky", true) : true;

        float bestScore = aimEnabled ? aimFovPx : (sw * 0.18f);

        for (const auto& p : world.players) {
            if (teamChk && p.isTeammate) continue;
            if (p.distanceM > maxRange) continue;

            Vec3 aimPt = SelectTargetBone(p, boneMode, world.vpMatrix, sw, sh);
            if (predict) {
                float travelT = std::clamp((p.distanceM / 650.0f + 0.012f) * leadMul, 0.0f, 0.09f);
                aimPt += p.velocity * travelT;
            }

            Vec2 sc{};
            if (!world.vpMatrix.WorldToScreen(aimPt, sw, sh, sc)) continue;
            float pxDist = (sc - Vec2(screenCenter.x, screenCenter.y)).Length();

            // Sticky hysteresis bonus for currently tracked operator
            float effectiveScore = (sticky && g_Combat.hasLockedTarget && p.address == g_Combat.lockedAddress)
                                   ? (pxDist * 0.72f) : pxDist;

            if (effectiveScore <= bestScore) {
                bestScore        = effectiveScore;
                bestTarget       = &p;
                bestTargetScreen = sc;
                bestAimWorld     = aimPt;
            }
        }
    }

    if (bestTarget) {
        g_Combat.hasLockedTarget = true;
        g_Combat.lockedAddress   = bestTarget->address;
        g_Combat.lockedSnapshot  = *bestTarget;
    } else {
        g_Combat.hasLockedTarget = false;
        g_Combat.lockedAddress   = 0;
    }

    // Active mouse tracking when game cursor is locked and menu/free-cursor are closed
    if (aimEnabled && bestTarget && !menuOpen && !freeCursorActive && world.cursorLocked) {
        int  actMode  = mAim->GetCombo("activation", 0);
        bool bindHeld = (mAim->keybind > 0) && ((GetAsyncKeyState(mAim->keybind) & 0x8000) != 0);
        bool activeTrigger = false;
        switch (actMode) {
            case 0: activeTrigger = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0; break;
            case 1: activeTrigger = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0; break;
            case 2: activeTrigger = ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0) && ((GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0); break;
            case 3: activeTrigger = true; break;
            case 4: activeTrigger = bindHeld; break;
        }
        if (mAim->bindMode == BindMode::Hold && bindHeld) {
            activeTrigger = true;
        }

        if (activeTrigger) {
            if (world.seq != g_Combat.lastAimSnapSeq) {
                g_Combat.lastAimSnapSeq = world.seq;
                g_Combat.snapSentDegX   = 0.0f;
                g_Combat.snapSentDegY   = 0.0f;
            }

            float smooth    = std::max(1.5f, mAim->GetFloat("smooth", 8.0f));
            float sensScale = mAim->GetFloat("sens_scale", 1.00f);
            float rcsPitch  = mAim->GetFloat("rcs_pitch", 18.0f) * 0.01f;
            float dt60      = std::clamp(dt * 60.0f, 0.05f, 2.5f);

            float dxPx = bestTargetScreen.x - screenCenter.x;
            float dyPx = bestTargetScreen.y - screenCenter.y;

            // Convert screen offset to true angular error in degrees & subtract counts already sent on this snapshot
            float rawYawDeg   = std::atan2(dxPx, std::max(100.0f, degToPx)) * (180.0f / std::numbers::pi_v<float>);
            float rawPitchDeg = std::atan2(dyPx, std::max(100.0f, degToPx)) * (180.0f / std::numbers::pi_v<float>);
            float yawDeg      = rawYawDeg   - g_Combat.snapSentDegX;
            float pitchDeg    = rawPitchDeg - g_Combat.snapSentDegY;
            if (yawDeg   * rawYawDeg   < 0.0f) yawDeg   = 0.0f;
            if (pitchDeg * rawPitchDeg < 0.0f) pitchDeg = 0.0f;

            float errDeg = std::hypot(yawDeg, pitchDeg);
            float errPx  = std::hypot(dxPx, dyPx);

            float rawDx = 0.0f;
            float rawDy = 0.0f;

            if (errDeg < 0.18f || errPx < 1.6f) {
                g_Combat.subPixelX = 0.0f;
                g_Combat.subPixelY = 0.0f;
            } else {
                // Smooth proportional-damped landing ramp inside 1.85 deg to eliminate center oscillation
                float nearDamp  = std::clamp((errDeg - 0.14f) / 1.85f, 0.20f, 1.0f);
                float stepScale = (2.25f * sensScale / smooth) * dt60 * nearDamp;

                rawDx = yawDeg   * stepScale;
                rawDy = pitchDeg * stepScale;

                if (g_Combat.subPixelX * rawDx < 0.0f) g_Combat.subPixelX = 0.0f;
                if (g_Combat.subPixelY * rawDy < 0.0f) g_Combat.subPixelY = 0.0f;
            }

            // Frame-rate independent downward pitch stabilization while firing LMB
            if ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0 && rcsPitch > 0.01f) {
                rawDy += (7.2f * rcsPitch) * dt;
            }

            g_Combat.subPixelX = std::clamp(g_Combat.subPixelX + std::clamp(rawDx, -12.0f, 12.0f), -14.0f, 14.0f);
            g_Combat.subPixelY = std::clamp(g_Combat.subPixelY + std::clamp(rawDy, -12.0f, 12.0f), -14.0f, 14.0f);

            int moveX = static_cast<int>(std::trunc(g_Combat.subPixelX));
            int moveY = static_cast<int>(std::trunc(g_Combat.subPixelY));
            if (moveX != 0 || moveY != 0) {
                g_Combat.subPixelX -= static_cast<float>(moveX);
                g_Combat.subPixelY -= static_cast<float>(moveY);
                constexpr float kEstDegPerCount = 0.095f;
                g_Combat.snapSentDegX += static_cast<float>(moveX) * kEstDegPerCount;
                g_Combat.snapSentDegY += static_cast<float>(moveY) * kEstDegPerCount;
                SendMouseDelta(moveX, moveY);
            }
        } else {
            g_Combat.subPixelX = g_Combat.subPixelY = 0.0f;
        }
    } else {
        g_Combat.subPixelX = g_Combat.subPixelY = 0.0f;
    }

    // ---------------- 2. 3D RAY-CAPSULE AUTO TRIGGER ----------------
    if (mTrig && mTrig->enabled && world.inMatch && !menuOpen && !freeCursorActive && world.cursorLocked) {
        int  tMode   = mTrig->GetCombo("mode", 0);
        bool keyOk   = (tMode == 0) || (mTrig->keybind > 0 && (GetAsyncKeyState(mTrig->keybind) & 0x8000) != 0);
        bool adsOk   = !mTrig->GetBool("ads_only", false) || ((GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0);
        bool teamChk = mTrig->GetBool("team_chk", true);
        float rad    = mTrig->GetFloat("radius", 0.20f);
        double delaySec   = mTrig->GetFloat("delay_ms", 25.0f) * 0.001;
        double cadenceSec = mTrig->GetFloat("cadence_ms", 95.0f) * 0.001;

        bool intersecting = false;
        if (keyOk && adsOk) {
            for (const auto& p : world.players) {
                if (teamChk && p.isTeammate) continue;
                const Vec3 testPts[4] = { p.bones.head, p.bones.neck, p.bones.chest, p.bones.pelvis };
                for (int bi = 0; bi < 4; ++bi) {
                    Vec3 toBone = testPts[bi] - world.camPos;
                    float along = toBone.Dot(world.camFwd);
                    if (along <= 0.2f || along > 260.0f) continue;
                    Vec3 closest = world.camPos + world.camFwd * along;
                    float perpDist = (testPts[bi] - closest).Length();
                    float boneRad = (bi == 0) ? (rad * 0.78f) : rad;
                    if (perpDist <= boneRad) {
                        intersecting = true;
                        break;
                    }
                }
                if (intersecting) break;
            }
        }

        if (intersecting) {
            if (g_Combat.firstSightTime <= 0.0) g_Combat.firstSightTime = nowSec;
            if ((nowSec - g_Combat.firstSightTime) >= delaySec && (nowSec - g_Combat.lastShotTime) >= cadenceSec) {
                if ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) == 0) {
                    TapMouseClick();
                    g_Combat.lastShotTime = nowSec;
                }
            }
        } else {
            g_Combat.firstSightTime = 0.0;
        }
    }

    // ---------------- 3. OPERATOR ESP & SNAPLINES ----------------
    bool espOn  = mEsp && mEsp->enabled && world.inMatch;
    bool snapOn = mSnap && mSnap->enabled && world.inMatch;
    float closestEnemyDist = 9999.0f;

    if (world.inMatch) {
        for (const auto& p : world.players) {
            if (!p.isTeammate) closestEnemyDist = std::min(closestEnemyDist, p.distanceM);

            bool isLocked = (g_Combat.hasLockedTarget && p.address == g_Combat.lockedAddress);

            // Snaplines
            if (snapOn && (!mSnap->GetBool("enemy_only", true) || !p.isTeammate)) {
                Vec2 footSc{};
                if (world.vpMatrix.WorldToScreen(p.feetPos, sw, sh, footSc)) {
                    int orig = mSnap->GetCombo("origin", 0);
                    ImVec2 p0 = (orig == 0) ? ImVec2(sw * 0.5f, sh - 10.0f)
                              : (orig == 1) ? screenCenter
                                            : ImVec2(sw * 0.5f, 12.0f);
                    ImVec2 p1(footSc.x, footSc.y);
                    ImU32 scol = ImGui::ColorConvertFloat4ToU32(isLocked ? ImVec4(1.0f, 0.84f, 0.30f, 0.90f)
                                                                         : mSnap->GetColor("col", pal.accent));
                    if (mSnap->GetBool("bezier", false)) {
                        ImVec2 ctrl1(p0.x, (p0.y + p1.y) * 0.5f);
                        ImVec2 ctrl2(p1.x, (p0.y + p1.y) * 0.5f);
                        bgList->AddBezierCubic(p0, ctrl1, ctrl2, p1, scol, 1.35f);
                    } else {
                        bgList->AddLine(p0, p1, scol, 1.25f);
                    }
                }
            }

            if (!espOn) continue;
            if (mEsp->GetBool("enemy_only", true) && p.isTeammate) continue;
            if (p.distanceM > mEsp->GetFloat("max_dist", 250.0f)) continue;

            ImVec4 boxColF = isLocked     ? mEsp->GetColor("col_locked", ImVec4(1.0f, 0.82f, 0.28f, 1.0f))
                           : p.isTeammate ? mEsp->GetColor("col_team",   ImVec4(0.28f, 0.78f, 0.96f, 1.0f))
                                          : mEsp->GetColor("col_enemy",  ImVec4(0.96f, 0.29f, 0.41f, 1.0f));
            ImU32 boxCol   = ImGui::ColorConvertFloat4ToU32(boxColF);
            ImU32 darkOut  = IM_COL32(0, 0, 0, 195);

            Vec3 topWorld = p.headPos + Vec3(0.0f, 0.20f, 0.0f);
            Vec3 botWorld = p.feetPos - Vec3(0.0f, 0.04f, 0.0f);
            Vec2 topSc{}, botSc{};
            bool onScreen = world.vpMatrix.WorldToScreen(topWorld, sw, sh, topSc) &&
                            world.vpMatrix.WorldToScreen(botWorld, sw, sh, botSc) &&
                            topSc.x >= -60.0f && topSc.x <= sw + 60.0f &&
                            botSc.y >= -60.0f && topSc.y <= sh + 60.0f;

            if (!onScreen) {
                // Offscreen directional arrows
                if (mEsp->GetBool("oof_arrows", true) && !p.isTeammate) {
                    float cx = 0.0f, cy = 0.0f, cw = 0.0f;
                    world.vpMatrix.WorldToRadarClip(p.bones.chest, cx, cy, cw);
                    if (cw < 0.0f) { cx = -cx; cy = -cy; }
                    float ang = std::atan2(-cy, cx);
                    float rad = mEsp->GetFloat("oof_radius", 185.0f);
                    ImVec2 tip(screenCenter.x + std::cos(ang) * rad, screenCenter.y + std::sin(ang) * rad);
                    ImVec2 left(tip.x - std::cos(ang - 0.42f) * 14.0f, tip.y - std::sin(ang - 0.42f) * 14.0f);
                    ImVec2 right(tip.x - std::cos(ang + 0.42f) * 14.0f, tip.y - std::sin(ang + 0.42f) * 14.0f);
                    float pulse = 0.65f + 0.35f * std::sin(static_cast<float>(nowSec) * 6.0f);
                    ImVec4 pc = boxColF; pc.w *= pulse;
                    bgList->AddTriangleFilled(tip, left, right, ImGui::ColorConvertFloat4ToU32(pc));
                    bgList->AddTriangle(tip, left, right, darkOut, 1.2f);
                }
                continue;
            }

            float boxH = std::fabs(botSc.y - topSc.y);
            if (boxH < 6.0f || boxH > sh * 1.6f) continue;
            float boxW = boxH * (p.isCrouched ? 0.62f : 0.48f);
            float cx   = (topSc.x + botSc.x) * 0.5f;
            ImVec2 bMin(cx - boxW * 0.5f, std::min(topSc.y, botSc.y));
            ImVec2 bMax(cx + boxW * 0.5f, std::max(topSc.y, botSc.y));

            int boxStyle = mEsp->GetCombo("box_mode", 0);
            if (mEsp->GetBool("box_fill", true) && boxStyle != 3) {
                ImU32 fillTop = ImGui::ColorConvertFloat4ToU32(ImVec4(boxColF.x, boxColF.y, boxColF.z, 0.06f));
                ImU32 fillBot = ImGui::ColorConvertFloat4ToU32(ImVec4(boxColF.x, boxColF.y, boxColF.z, 0.17f));
                bgList->AddRectFilledMultiColor(bMin, bMax, fillTop, fillTop, fillBot, fillBot);
            }

            if (boxStyle == 0) {
                // CS2 crisp corner brackets
                float cw = std::max(4.0f, boxW * 0.26f);
                float ch = std::max(4.0f, boxH * 0.22f);
                auto drawCorner = [&](ImVec2 o, float dx, float dy, ImU32 col, float th) {
                    bgList->AddLine(o, ImVec2(o.x + dx, o.y), col, th);
                    bgList->AddLine(o, ImVec2(o.x, o.y + dy), col, th);
                };
                drawCorner(bMin, cw, ch, darkOut, 3.0f);
                drawCorner(ImVec2(bMax.x, bMin.y), -cw, ch, darkOut, 3.0f);
                drawCorner(ImVec2(bMin.x, bMax.y), cw, -ch, darkOut, 3.0f);
                drawCorner(bMax, -cw, -ch, darkOut, 3.0f);

                drawCorner(bMin, cw, ch, boxCol, 1.5f);
                drawCorner(ImVec2(bMax.x, bMin.y), -cw, ch, boxCol, 1.5f);
                drawCorner(ImVec2(bMin.x, bMax.y), cw, -ch, boxCol, 1.5f);
                drawCorner(bMax, -cw, -ch, boxCol, 1.5f);
            } else if (boxStyle == 1) {
                bgList->AddRect(ImVec2(bMin.x - 1.0f, bMin.y - 1.0f), ImVec2(bMax.x + 1.0f, bMax.y + 1.0f), darkOut, 2.0f, 0, 1.0f);
                bgList->AddRect(bMin, bMax, boxCol, 2.0f, 0, 1.4f);
            } else if (boxStyle == 2) {
                // 3D World-Space Bounding Prism
                float hw = 0.36f;
                Vec3 c3[8] = {
                    { p.feetPos.x - hw, botWorld.y, p.feetPos.z - hw },
                    { p.feetPos.x + hw, botWorld.y, p.feetPos.z - hw },
                    { p.feetPos.x + hw, botWorld.y, p.feetPos.z + hw },
                    { p.feetPos.x - hw, botWorld.y, p.feetPos.z + hw },
                    { p.feetPos.x - hw, topWorld.y, p.feetPos.z - hw },
                    { p.feetPos.x + hw, topWorld.y, p.feetPos.z - hw },
                    { p.feetPos.x + hw, topWorld.y, p.feetPos.z + hw },
                    { p.feetPos.x - hw, topWorld.y, p.feetPos.z + hw }
                };
                Vec2 s2[8]{};
                bool ok8 = true;
                for (int k = 0; k < 8; ++k) {
                    if (!world.vpMatrix.WorldToScreen(c3[k], sw, sh, s2[k])) { ok8 = false; break; }
                }
                if (ok8) {
                    static const int edges[12][2] = {
                        {0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}
                    };
                    for (auto& e : edges) {
                        bgList->AddLine(ImVec2(s2[e[0]].x, s2[e[0]].y), ImVec2(s2[e[1]].x, s2[e[1]].y), darkOut, 2.4f);
                        bgList->AddLine(ImVec2(s2[e[0]].x, s2[e[0]].y), ImVec2(s2[e[1]].x, s2[e[1]].y), boxCol, 1.2f);
                    }
                }
            }

            // Articulated Bone Skeleton
            if (mEsp->GetBool("skeleton", true)) {
                ImU32 skCol = ImGui::ColorConvertFloat4ToU32(mEsp->GetColor("col_bones", ImVec4(0.96f, 0.96f, 0.99f, 0.90f)));
                auto boneSeg = [&](const Vec3& a, const Vec3& b) {
                    Vec2 sa{}, sb{};
                    if (world.vpMatrix.WorldToScreen(a, sw, sh, sa) && world.vpMatrix.WorldToScreen(b, sw, sh, sb)) {
                        bgList->AddLine(ImVec2(sa.x, sa.y), ImVec2(sb.x, sb.y), IM_COL32(0, 0, 0, 150), 2.5f);
                        bgList->AddLine(ImVec2(sa.x, sa.y), ImVec2(sb.x, sb.y), skCol, 1.25f);
                    }
                };
                boneSeg(p.bones.head,   p.bones.neck);
                boneSeg(p.bones.neck,   p.bones.chest);
                boneSeg(p.bones.chest,  p.bones.pelvis);
                boneSeg(p.bones.pelvis, p.bones.lKnee);
                boneSeg(p.bones.lKnee,  p.bones.lFoot);
                boneSeg(p.bones.pelvis, p.bones.rKnee);
                boneSeg(p.bones.rKnee,  p.bones.rFoot);
                boneSeg(p.bones.neck,   p.bones.rHand);
                boneSeg(p.bones.neck,   p.bones.lHand);
            }

            if (mEsp->GetBool("head_dot", true)) {
                Vec2 hs{};
                if (world.vpMatrix.WorldToScreen(p.bones.head, sw, sh, hs)) {
                    float hr = std::clamp(boxW * 0.11f, 2.2f, 6.5f);
                    bgList->AddCircle(ImVec2(hs.x, hs.y), hr + 0.8f, darkOut, 20, 1.8f);
                    bgList->AddCircle(ImVec2(hs.x, hs.y), hr, boxCol, 20, 1.3f);
                }
            }

            // CS2 Vertical Health Bar (Left of Box)
            if (mEsp->GetBool("hp_bar", true)) {
                float frac = std::clamp(p.health / std::max(1.0f, p.maxHealth), 0.0f, 1.0f);
                float bx = bMin.x - 6.0f;
                bgList->AddRectFilled(ImVec2(bx - 1.5f, bMin.y - 1.0f), ImVec2(bx + 2.5f, bMax.y + 1.0f), IM_COL32(8, 8, 10, 210), 1.5f);
                ImVec4 hpCol(1.0f - frac * 0.72f, 0.24f + frac * 0.72f, 0.32f, 1.0f);
                float fillTopY = bMax.y - (bMax.y - bMin.y) * frac;
                bgList->AddRectFilled(ImVec2(bx - 0.5f, fillTopY), ImVec2(bx + 1.5f, bMax.y), ImGui::ColorConvertFloat4ToU32(hpCol), 1.0f);

                if (p.health < 99.5f && g_Fonts.espBold13) {
                    char hpTxt[16]{};
                    std::snprintf(hpTxt, sizeof(hpTxt), "%d", static_cast<int>(std::round(p.health)));
                    ImVec2 ts = g_Fonts.espBold13->CalcTextSizeA(11.5f, FLT_MAX, 0.0f, hpTxt);
                    bgList->AddText(g_Fonts.espBold13, 11.5f, ImVec2(bx - ts.x - 2.5f + 1.0f, fillTopY - 5.0f + 1.0f), darkOut, hpTxt);
                    bgList->AddText(g_Fonts.espBold13, 11.5f, ImVec2(bx - ts.x - 2.5f, fillTopY - 5.0f), IM_COL32(255, 255, 255, 240), hpTxt);
                }
            }

            // CS2 Armour Bar (Bottom of Box when armour > 0)
            float bottomTagOffset = 3.0f;
            if (mEsp->GetBool("armor_bar", true) && p.armor > 1.0f) {
                float afrac = std::clamp(p.armor / 100.0f, 0.0f, 1.0f);
                float ay = bMax.y + 3.5f;
                bgList->AddRectFilled(ImVec2(bMin.x - 1.0f, ay - 1.0f), ImVec2(bMax.x + 1.0f, ay + 3.0f), IM_COL32(8, 8, 10, 210), 1.5f);
                bgList->AddRectFilled(ImVec2(bMin.x, ay), ImVec2(bMin.x + (bMax.x - bMin.x) * afrac, ay + 2.0f), IM_COL32(72, 186, 255, 245), 1.0f);
                bottomTagOffset = 8.5f;
            }

            // Operator Name Tag (Top Center)
            if (mEsp->GetBool("name_tag", true) && g_Fonts.espBold13) {
                std::string label = p.name;
                if (p.isBot) label += " [BOT]";
                ImVec2 ts = g_Fonts.espBold13->CalcTextSizeA(13.0f, FLT_MAX, 0.0f, label.c_str());
                ImVec2 tp(cx - ts.x * 0.5f, bMin.y - ts.y - 3.0f);
                bgList->AddText(g_Fonts.espBold13, 13.0f, ImVec2(tp.x + 1.0f, tp.y + 1.0f), darkOut, label.c_str());
                bgList->AddText(g_Fonts.espBold13, 13.0f, tp, isLocked ? IM_COL32(255, 218, 92, 255) : IM_COL32(246, 247, 252, 250), label.c_str());
            }

            // Weapon & Distance Tags (Bottom Center)
            if ((mEsp->GetBool("weap_tag", true) || mEsp->GetBool("dist_tag", true)) && g_Fonts.esp13) {
                std::string sub;
                if (mEsp->GetBool("weap_tag", true)) sub = p.weaponName;
                if (mEsp->GetBool("dist_tag", true)) {
                    char dbuf[24]{};
                    std::snprintf(dbuf, sizeof(dbuf), "%.0fm", p.distanceM);
                    if (!sub.empty()) sub += "  ";
                    sub += dbuf;
                }
                ImVec2 ts = g_Fonts.esp13->CalcTextSizeA(12.0f, FLT_MAX, 0.0f, sub.c_str());
                ImVec2 tp(cx - ts.x * 0.5f, bMax.y + bottomTagOffset);
                bgList->AddText(g_Fonts.esp13, 12.0f, ImVec2(tp.x + 1.0f, tp.y + 1.0f), darkOut, sub.c_str());
                bgList->AddText(g_Fonts.esp13, 12.0f, tp, IM_COL32(205, 210, 224, 235), sub.c_str());
            }
        }
    }

    // ---------------- 4. S&D PLANTED C4 WORLD PIN & COUNTDOWN BAR ----------------
    if (mBomb && mBomb->enabled && world.bomb.active) {
        float tLeft = world.bomb.timeLeft;
        bool critical = (tLeft < 10.0f);
        bool noKitFail = (tLeft < 5.0f);
        ImVec4 c4Col = noKitFail ? ImVec4(1.0f, 0.20f, 0.25f, 1.0f)
                     : critical  ? ImVec4(1.0f, 0.58f, 0.18f, 1.0f)
                                 : ImVec4(0.96f, 0.82f, 0.28f, 1.0f);

        if (mBomb->GetBool("world_pin", true)) {
            Vec2 bs{};
            if (world.vpMatrix.WorldToScreen(world.bomb.position + Vec3(0.0f, 0.25f, 0.0f), sw, sh, bs)) {
                float pulse = 14.0f + 4.0f * std::sin(static_cast<float>(nowSec) * 8.0f);
                bgList->AddCircle(ImVec2(bs.x, bs.y), pulse, ImGui::ColorConvertFloat4ToU32(ImVec4(c4Col.x, c4Col.y, c4Col.z, 0.45f)), 32, 2.0f);
                bgList->AddCircleFilled(ImVec2(bs.x, bs.y), 10.0f, IM_COL32(14, 14, 18, 225), 24);
                char cbuf[32]{};
                std::snprintf(cbuf, sizeof(cbuf), "C4  %.1fs", tLeft);
                if (g_Fonts.espBold13) {
                    ImVec2 ts = g_Fonts.espBold13->CalcTextSizeA(13.5f, FLT_MAX, 0.0f, cbuf);
                    bgList->AddText(g_Fonts.espBold13, 13.5f, ImVec2(bs.x - ts.x * 0.5f, bs.y + 13.0f), ImGui::ColorConvertFloat4ToU32(c4Col), cbuf);
                }
            }
        }

        if (mBomb->GetBool("top_bar", true)) {
            float bw = 310.0f, bh = 38.0f;
            ImVec2 pMin(sw * 0.5f - bw * 0.5f, 52.0f);
            ImVec2 pMax(sw * 0.5f + bw * 0.5f, 52.0f + bh);
            bgList->AddRectFilled(pMin, pMax, IM_COL32(14, 14, 18, 232), 8.0f);
            bgList->AddRect(pMin, pMax, ImGui::ColorConvertFloat4ToU32(ImVec4(c4Col.x, c4Col.y, c4Col.z, 0.55f)), 8.0f, 0, 1.2f);

            float frac = std::clamp(tLeft / std::max(1.0f, world.bomb.maxTime), 0.0f, 1.0f);
            bgList->AddRectFilled(ImVec2(pMin.x + 10.0f, pMax.y - 8.0f), ImVec2(pMax.x - 10.0f, pMax.y - 4.0f), IM_COL32(36, 36, 44, 255), 2.0f);
            bgList->AddRectFilled(ImVec2(pMin.x + 10.0f, pMax.y - 8.0f), ImVec2(pMin.x + 10.0f + (bw - 20.0f) * frac, pMax.y - 4.0f), ImGui::ColorConvertFloat4ToU32(c4Col), 2.0f);

            float distB = (world.localPlayerPtr ? world.localPos : world.camPos).Dist(world.bomb.position);
            int estDmg  = static_cast<int>(std::max(0.0f, 500.0f * std::exp(-(distB * distB) / (2.0f * 16.5f * 16.5f))));
            char hdr[64]{};
            if (mBomb->GetBool("dmg_calc", true)) {
                std::snprintf(hdr, sizeof(hdr), ICON_FA_FLAME "  C4 PLANTED: %.1fs   |   Blast: ~%d HP", tLeft, estDmg);
            } else {
                std::snprintf(hdr, sizeof(hdr), ICON_FA_FLAME "  C4 PLANTED: %.1fs", tLeft);
            }
            if (g_Fonts.bold16) {
                ImVec2 ts = g_Fonts.bold16->CalcTextSizeA(14.0f, FLT_MAX, 0.0f, hdr);
                bgList->AddText(g_Fonts.bold16, 14.0f, ImVec2(sw * 0.5f - ts.x * 0.5f, pMin.y + 7.0f), IM_COL32(250, 250, 255, 255), hdr);
            }
        }
    }

    // ---------------- 5. PROXIMITY PERIMETER ALERT ----------------
    if (mProx && mProx->enabled && world.inMatch) {
        float warnR = mProx->GetFloat("radius", 18.0f);
        if (closestEnemyDist <= warnR) {
            if (mProx->GetBool("chime", true) && (nowSec - g_Combat.lastProxChime) > 2.2) {
                g_Sound.PlayAlert();
                g_Combat.lastProxChime = nowSec;
            }
            char wbuf[64]{};
            std::snprintf(wbuf, sizeof(wbuf), ICON_FA_BOLT "  HOSTILE OPERATOR CLOSE  (%.1f m)", closestEnemyDist);
            ImFont* f = g_Fonts.bold16 ? g_Fonts.bold16 : ImGui::GetFont();
            ImVec2 ts = f->CalcTextSizeA(14.5f, FLT_MAX, 0.0f, wbuf);
            ImVec2 p0(sw * 0.5f - ts.x * 0.5f - 16.0f, sh * 0.16f);
            ImVec2 p1(sw * 0.5f + ts.x * 0.5f + 16.0f, sh * 0.16f + 30.0f);
            float pulse = 0.65f + 0.35f * std::sin(static_cast<float>(nowSec) * 9.0f);
            bgList->AddRectFilled(p0, p1, IM_COL32(20, 12, 16, 225), 7.0f);
            bgList->AddRect(p0, p1, ImGui::ColorConvertFloat4ToU32(ImVec4(0.98f, 0.28f, 0.38f, pulse)), 7.0f, 0, 1.5f);
            bgList->AddText(f, 14.5f, ImVec2(p0.x + 16.0f, p0.y + 6.5f), IM_COL32(255, 232, 236, 255), wbuf);
        }
    }

    // ---------------- 6. TACTICAL CROSSHAIR ----------------
    if (mCross && mCross->enabled) {
        int   style = mCross->GetCombo("style", 0);
        float len   = mCross->GetFloat("length", 6.5f);
        float gap   = mCross->GetFloat("gap", 3.5f);
        float thick = mCross->GetFloat("thick", 1.5f);
        bool  dyn   = mCross->GetBool("dynamic", true);
        bool  outl  = mCross->GetBool("outline", true);

        float extraGap = 0.0f;
        if (dyn) {
            bool firing = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
            float targetExpand = (firing ? 4.5f : 0.0f) + (g_Combat.hasLockedTarget ? 2.0f : 0.0f);
            extraGap = g_Anim.StepKey("hud.cross_gap", targetExpand, dt);
        }
        float g = gap + extraGap;
        ImU32 col = ImGui::ColorConvertFloat4ToU32(mCross->GetColor("col", ImVec4(0.30f, 0.98f, 0.62f, 0.95f)));
        ImU32 ocol = IM_COL32(0, 0, 0, 200);

        auto arm = [&](ImVec2 a, ImVec2 b) {
            if (outl) bgList->AddLine(a, b, ocol, thick + 2.0f);
            bgList->AddLine(a, b, col, thick);
        };

        if (style == 0 || style == 3) {
            if (style == 0) arm(ImVec2(screenCenter.x, screenCenter.y - g - len), ImVec2(screenCenter.x, screenCenter.y - g));
            arm(ImVec2(screenCenter.x, screenCenter.y + g), ImVec2(screenCenter.x, screenCenter.y + g + len));
            arm(ImVec2(screenCenter.x - g - len, screenCenter.y), ImVec2(screenCenter.x - g, screenCenter.y));
            arm(ImVec2(screenCenter.x + g, screenCenter.y), ImVec2(screenCenter.x + g + len, screenCenter.y));
        } else if (style == 1) {
            if (outl) bgList->AddCircle(screenCenter, g + len * 0.65f, ocol, 32, thick + 1.8f);
            bgList->AddCircle(screenCenter, g + len * 0.65f, col, 32, thick);
            bgList->AddCircleFilled(screenCenter, thick * 1.1f, col, 12);
        } else if (style == 2) {
            if (outl) bgList->AddCircleFilled(screenCenter, thick + 1.6f, ocol, 16);
            bgList->AddCircleFilled(screenCenter, thick + 0.5f, col, 16);
        }
    }

    // ---------------- 7. DRAGGABLE 2D RADAR ----------------
    if (mRadar && mRadar->enabled) {
        float rSize  = mRadar->GetFloat("size", 190.0f);
        float rRange = std::max(10.0f, mRadar->GetFloat("range", 75.0f));
        bool  circle = (mRadar->GetCombo("shape", 0) == 1);

        if (menuOpen || freeCursorActive) {
            ImGui::SetNextWindowPos(g_Combat.radarPos, ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(rSize, rSize));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            if (ImGui::Begin("##RelinquishRadarDrag", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings)) {
                g_Combat.radarPos = ImGui::GetWindowPos();
            }
            ImGui::End();
            ImGui::PopStyleVar();
        }

        ImVec2 rMin = g_Combat.radarPos;
        ImVec2 rMax(rMin.x + rSize, rMin.y + rSize);
        ImVec2 rCen(rMin.x + rSize * 0.5f, rMin.y + rSize * 0.5f);
        float  half = rSize * 0.5f;

        if (circle) {
            bgList->AddCircleFilled(rCen, half, IM_COL32(14, 14, 18, 224), 64);
            bgList->AddCircle(rCen, half, ImGui::ColorConvertFloat4ToU32(pal.border), 64, 1.3f);
            bgList->AddCircle(rCen, half * 0.5f, IM_COL32(255, 255, 255, 16), 48, 1.0f);
        } else {
            bgList->AddRectFilled(rMin, rMax, IM_COL32(14, 14, 18, 224), 10.0f);
            bgList->AddRect(rMin, rMax, ImGui::ColorConvertFloat4ToU32(pal.border), 10.0f, 0, 1.3f);
            bgList->AddRect(ImVec2(rCen.x - half * 0.5f, rCen.y - half * 0.5f), ImVec2(rCen.x + half * 0.5f, rCen.y + half * 0.5f), IM_COL32(255, 255, 255, 14), 5.0f);
        }

        bgList->AddLine(ImVec2(rMin.x + 8.0f, rCen.y), ImVec2(rMax.x - 8.0f, rCen.y), IM_COL32(255, 255, 255, 18), 1.0f);
        bgList->AddLine(ImVec2(rCen.x, rMin.y + 8.0f), ImVec2(rCen.x, rMax.y - 8.0f), IM_COL32(255, 255, 255, 18), 1.0f);

        if (mRadar->GetBool("fov_cone", true)) {
            float halfFovRad = std::clamp(world.camFovDeg, 40.0f, 120.0f) * 0.5f * (std::numbers::pi_v<float> / 180.0f);
            ImVec2 fL(rCen.x - std::sin(halfFovRad) * (half - 8.0f), rCen.y - std::cos(halfFovRad) * (half - 8.0f));
            ImVec2 fR(rCen.x + std::sin(halfFovRad) * (half - 8.0f), rCen.y - std::cos(halfFovRad) * (half - 8.0f));
            bgList->AddTriangleFilled(rCen, fL, fR, ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, 0.12f)));
            bgList->AddLine(rCen, fL, ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, 0.36f)), 1.0f);
            bgList->AddLine(rCen, fR, ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, 0.36f)), 1.0f);
        }

        // Camera horizontal basis for view-aligned radar rotation
        Vec3 flatFwd(world.camFwd.x, 0.0f, world.camFwd.z);
        if (flatFwd.Length() < 1e-4f) flatFwd = Vec3(0.0f, 0.0f, 1.0f);
        flatFwd = flatFwd.Normalized();
        Vec3 flatRight(flatFwd.z, 0.0f, -flatFwd.x); // Unity left-handed XZ right vector: check via camRight!
        if (flatRight.Dot(Vec3(world.camRight.x, 0.0f, world.camRight.z)) < 0.0f) {
            flatRight = flatRight * -1.0f;
        }
        Vec3 origin = world.localPlayerPtr ? world.localPos : world.camPos;

        for (const auto& p : world.players) {
            if (mRadar->GetBool("enemy_only", false) && p.isTeammate) continue;
            Vec3 rel = p.feetPos - origin;
            float rx = (rel.Dot(flatRight) / rRange) * (half - 10.0f);
            float ry = -(rel.Dot(flatFwd)  / rRange) * (half - 10.0f);
            ImVec2 blip(rCen.x + rx, rCen.y + ry);
            if (circle) {
                Vec2 d(blip.x - rCen.x, blip.y - rCen.y);
                if (d.Length() > half - 8.0f) {
                    d = d * ((half - 8.0f) / d.Length());
                    blip = ImVec2(rCen.x + d.x, rCen.y + d.y);
                }
            } else {
                blip.x = std::clamp(blip.x, rMin.x + 7.0f, rMax.x - 7.0f);
                blip.y = std::clamp(blip.y, rMin.y + 7.0f, rMax.y - 7.0f);
            }
            bool locked = (g_Combat.hasLockedTarget && p.address == g_Combat.lockedAddress);
            ImU32 bcol = locked       ? IM_COL32(255, 214, 72, 255)
                       : p.isTeammate ? IM_COL32(72, 200, 255, 245)
                                      : IM_COL32(248, 74, 104, 255);
            bgList->AddCircleFilled(blip, locked ? 5.0f : 4.0f, IM_COL32(0, 0, 0, 200), 16);
            bgList->AddCircleFilled(blip, locked ? 3.8f : 3.0f, bcol, 16);
        }

        if (world.bomb.active) {
            Vec3 rel = world.bomb.position - origin;
            float rx = std::clamp((rel.Dot(flatRight) / rRange) * (half - 10.0f), -(half - 8.0f), half - 8.0f);
            float ry = std::clamp(-(rel.Dot(flatFwd)  / rRange) * (half - 10.0f), -(half - 8.0f), half - 8.0f);
            bgList->AddCircleFilled(ImVec2(rCen.x + rx, rCen.y + ry), 4.8f, IM_COL32(255, 196, 48, 255), 16);
        }

        bgList->AddCircleFilled(rCen, 3.6f, IM_COL32(255, 255, 255, 255), 16);
    }

    // ---------------- 8. CS2 / RUSHER DRAGGABLE TARGET HUD ----------------
    if (mThud && mThud->enabled) {
        if (g_Combat.targetHudPos.x < 0.0f) {
            g_Combat.targetHudPos = ImVec2(sw * 0.5f + 36.0f, sh * 0.5f + 28.0f);
        }
        bool showPreview = (menuOpen || freeCursorActive) && !g_Combat.hasLockedTarget;
        float visAlpha = g_Anim.StepKey("hud.thud_alpha", (g_Combat.hasLockedTarget || showPreview) ? 1.0f : 0.0f, dt);

        if (visAlpha > 0.02f) {
            OperatorEntity disp = g_Combat.hasLockedTarget ? g_Combat.lockedSnapshot : OperatorEntity{};
            if (showPreview) {
                disp.name       = "Preview Operator";
                disp.weaponName = "AK-47 Vulcan";
                disp.health     = 78.0f;
                disp.maxHealth  = 100.0f;
                disp.armor      = 65.0f;
                disp.distanceM  = 24.5f;
            }

            bool compact = (mThud->GetCombo("style", 0) == 1);
            float cw = compact ? 225.0f : 262.0f;
            float ch = compact ? 64.0f  : 78.0f;

            if (menuOpen || freeCursorActive) {
                ImGui::SetNextWindowPos(g_Combat.targetHudPos, ImGuiCond_Once);
                ImGui::SetNextWindowSize(ImVec2(cw, ch));
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
                if (ImGui::Begin("##RelinquishTargetHudDrag", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings)) {
                    g_Combat.targetHudPos = ImGui::GetWindowPos();
                }
                ImGui::End();
                ImGui::PopStyleVar();
            }

            float hpFrac    = std::clamp(disp.health / std::max(1.0f, disp.maxHealth), 0.0f, 1.0f);
            float armorFrac = std::clamp(disp.armor / 100.0f, 0.0f, 1.0f);
            g_Combat.hpSmooth    = g_Anim.StepKey("hud.thud_hp", hpFrac, dt);
            g_Combat.armorSmooth = g_Anim.StepKey("hud.thud_ar", armorFrac, dt);
            if (hpFrac < g_Combat.hpGhostTrail) {
                g_Combat.hpGhostTrail = std::max(hpFrac, g_Combat.hpGhostTrail - dt * 0.45f);
            } else {
                g_Combat.hpGhostTrail = hpFrac;
            }

            ImVec2 p0 = g_Combat.targetHudPos;
            ImVec2 p1(p0.x + cw, p0.y + ch);
            bgList->AddRectFilled(p0, p1, IM_COL32(15, 15, 19, static_cast<int>(232 * visAlpha)), 11.0f);
            bgList->AddRect(p0, p1, ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, 0.45f * visAlpha)), 11.0f, 0, 1.2f);

            float textLeft = p0.x + 14.0f;
            if (!compact && mThud->GetBool("silhouette", true)) {
                ImVec2 av0(p0.x + 11.0f, p0.y + 11.0f);
                ImVec2 av1(p0.x + 55.0f, p1.y - 11.0f);
                bgList->AddRectFilled(av0, av1, IM_COL32(26, 26, 33, static_cast<int>(255 * visAlpha)), 8.0f);
                ImVec2 ac((av0.x + av1.x) * 0.5f, (av0.y + av1.y) * 0.5f);
                // Crisp procedural operator bust silhouette + headshot reticle
                bgList->AddCircleFilled(ImVec2(ac.x, ac.y - 8.5f), 7.5f, ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, 0.88f * visAlpha)), 20);
                bgList->AddRectFilled(ImVec2(ac.x - 12.0f, ac.y + 2.0f), ImVec2(ac.x + 12.0f, av1.y - 5.0f), IM_COL32(215, 218, 230, static_cast<int>(215 * visAlpha)), 6.0f);
                textLeft = av1.x + 11.0f;
            }

            ImFont* fName = g_Fonts.bold16 ? g_Fonts.bold16 : ImGui::GetFont();
            ImFont* fSub  = g_Fonts.reg14  ? g_Fonts.reg14  : ImGui::GetFont();
            bgList->AddText(fName, 14.5f, ImVec2(textLeft, p0.y + 10.0f), IM_COL32(248, 249, 253, static_cast<int>(255 * visAlpha)), disp.name.c_str());

            char meta[64]{};
            std::snprintf(meta, sizeof(meta), "%s  |  %.0f HP  |  %.1fm", disp.weaponName.c_str(), disp.health, disp.distanceM);
            bgList->AddText(fSub, 12.0f, ImVec2(textLeft, p0.y + 28.0f), IM_COL32(170, 175, 190, static_cast<int>(240 * visAlpha)), meta);

            // Animated HP bar + delayed ghost damage bar
            float barRight = p1.x - 12.0f;
            float barW     = std::max(20.0f, barRight - textLeft);
            float hpY      = p0.y + (compact ? 47.0f : 49.0f);
            bgList->AddRectFilled(ImVec2(textLeft, hpY), ImVec2(barRight, hpY + 7.0f), IM_COL32(32, 32, 40, static_cast<int>(240 * visAlpha)), 3.5f);
            bgList->AddRectFilled(ImVec2(textLeft, hpY), ImVec2(textLeft + barW * g_Combat.hpGhostTrail, hpY + 7.0f), IM_COL32(255, 196, 92, static_cast<int>(165 * visAlpha)), 3.5f);
            bgList->AddRectFilledMultiColor(
                ImVec2(textLeft, hpY), ImVec2(textLeft + barW * g_Combat.hpSmooth, hpY + 7.0f),
                ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, visAlpha)),
                ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accentSecondary.x, pal.accentSecondary.y, pal.accentSecondary.z, visAlpha)),
                ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accentSecondary.x, pal.accentSecondary.y, pal.accentSecondary.z, visAlpha)),
                ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, visAlpha))
            );

            if (!compact && mThud->GetBool("armor_bar", true)) {
                float arY = hpY + 11.0f;
                bgList->AddRectFilled(ImVec2(textLeft, arY), ImVec2(barRight, arY + 4.5f), IM_COL32(28, 32, 42, static_cast<int>(230 * visAlpha)), 2.2f);
                bgList->AddRectFilled(ImVec2(textLeft, arY), ImVec2(textLeft + barW * g_Combat.armorSmooth, arY + 4.5f), IM_COL32(76, 190, 255, static_cast<int>(240 * visAlpha)), 2.2f);
            }
        }
    }

    // ---------------- 9. DRAGGABLE TELEMETRY HUD ----------------
    if (mTele && mTele->enabled) {
        float tw = 228.0f;
        float th = 84.0f;
        if (menuOpen || freeCursorActive) {
            ImGui::SetNextWindowPos(g_Combat.telemetryPos, ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(tw, th));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            if (ImGui::Begin("##RelinquishTelemetryDrag", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings)) {
                g_Combat.telemetryPos = ImGui::GetWindowPos();
            }
            ImGui::End();
            ImGui::PopStyleVar();
        }

        ImVec2 p0 = g_Combat.telemetryPos;
        ImVec2 p1(p0.x + tw, p0.y + th);
        bgList->AddRectFilled(p0, p1, IM_COL32(14, 14, 18, 224), 10.0f);
        bgList->AddRect(p0, p1, ImGui::ColorConvertFloat4ToU32(pal.border), 10.0f, 0, 1.1f);

        ImFont* fb = g_Fonts.bold16 ? g_Fonts.bold16 : ImGui::GetFont();
        ImFont* fr = g_Fonts.reg14  ? g_Fonts.reg14  : ImGui::GetFont();

        char l1[64]{}, l2[80]{}, l3[80]{};
        if (world.attached) {
            std::snprintf(l1, sizeof(l1), ICON_FA_LINK "  CombatMaster.exe (PID %lu)", static_cast<unsigned long>(world.pid));
        } else {
            std::snprintf(l1, sizeof(l1), ICON_FA_ROTATE "  Awaiting CombatMaster.exe...");
        }
        Vec3 pos = world.localPlayerPtr ? world.localPos : world.camPos;
        std::snprintf(l2, sizeof(l2), "XYZ: %.1f, %.1f, %.1f  (%.1f m/s)", pos.x, pos.y, pos.z, world.localVel.Length2D());
        std::snprintf(l3, sizeof(l3), "Operators: %zu   FOV: %.0f   RPM/s: %llu",
                      world.players.size(), world.camFovDeg, static_cast<unsigned long long>(world.rpmPerSec));

        bgList->AddText(fb, 13.5f, ImVec2(p0.x + 12.0f, p0.y + 9.0f),
                        world.attached ? IM_COL32(98, 238, 162, 255) : IM_COL32(255, 184, 88, 255), l1);
        if (mTele->GetBool("coords", true)) {
            bgList->AddText(fr, 12.5f, ImVec2(p0.x + 12.0f, p0.y + 32.0f), IM_COL32(215, 218, 228, 240), l2);
        }
        if (mTele->GetBool("rpm", true)) {
            bgList->AddText(fr, 12.5f, ImVec2(p0.x + 12.0f, p0.y + 52.0f), IM_COL32(162, 168, 185, 240), l3);
        }
    }
}

// ============================================================================
// 9. CLIENT HUD: WATERMARK CREST, STEPPED ARRAYLIST & SLIDE-IN TOASTS
// ============================================================================
static void RenderClientHudAndToasts(ImDrawList* fgList, const WorldSnapshot& world, const Palette& pal,
                                     float sw, float sh, float dt) {
    Module* mHud = FindModule("client_hud");
    bool hudOn   = mHud && mHud->enabled;

    // 1. Top-Left RELINQUISH Emblem Watermark Pill
    if (hudOn && mHud->GetBool("watermark", true)) {
        float fps = ImGui::GetIO().Framerate;
        float ms  = (fps > 1.0f) ? (1000.0f / fps) : 0.0f;
        char wtxt[128]{};
        std::snprintf(wtxt, sizeof(wtxt), "RELINQUISH   |   %s   |   %.0f FPS  (%.2f ms)",
                      world.attached ? "CM x64 Attached" : "Standby", fps, ms);

        ImFont* f = g_Fonts.med15 ? g_Fonts.med15 : ImGui::GetFont();
        ImVec2 ts = f->CalcTextSizeA(14.0f, FLT_MAX, 0.0f, wtxt);
        float logoW = g_BrandLogo.srv ? 22.0f : 0.0f;
        float pw = ts.x + 28.0f + logoW;
        float ph = 32.0f;
        ImVec2 p0(18.0f, 16.0f);
        ImVec2 p1(18.0f + pw, 16.0f + ph);

        fgList->AddRectFilled(p0, p1, IM_COL32(14, 14, 18, 232), 8.0f);
        fgList->AddRect(p0, p1, ImGui::ColorConvertFloat4ToU32(pal.border), 8.0f, 0, 1.0f);
        fgList->AddRectFilledMultiColor(
            ImVec2(p0.x + 6.0f, p0.y), ImVec2(p1.x - 6.0f, p0.y + 2.0f),
            ImGui::ColorConvertFloat4ToU32(pal.accent),
            ImGui::ColorConvertFloat4ToU32(pal.accentSecondary),
            ImGui::ColorConvertFloat4ToU32(pal.accentSecondary),
            ImGui::ColorConvertFloat4ToU32(pal.accent)
        );

        float tx = p0.x + 12.0f;
        if (g_BrandLogo.srv) {
            fgList->AddImage(reinterpret_cast<ImTextureID>(g_BrandLogo.srv),
                             ImVec2(tx, p0.y + 5.0f), ImVec2(tx + 20.0f, p0.y + 27.0f),
                             ImVec2(0.38f, 0.02f), ImVec2(0.62f, 0.43f));
            tx += 25.0f;
        }
        fgList->AddText(f, 14.0f, ImVec2(tx, p0.y + (ph - ts.y) * 0.5f), IM_COL32(244, 245, 250, 250), wtxt);
    }

    // 2. Stepped Module Arraylist (1:1 Cloudy 1.3 ribbon aesthetic)
    if (hudOn && mHud->GetBool("arraylist", true)) {
        bool rightSide = (mHud->GetCombo("side", 0) == 0);
        ImFont* f = g_Fonts.med15 ? g_Fonts.med15 : ImGui::GetFont();

        struct ArrayEntry {
            const Module* mod;
            std::string   badge;
            float         width;
            float         animVis;
        };
        std::vector<ArrayEntry> active;
        for (const auto& m : g_Modules) {
            if (m.id == "client_hud") continue;
            float vis = g_Anim.Step(ImHashStr(("arr." + m.id).c_str()), m.enabled ? 1.0f : 0.0f, dt);
            if (vis <= 0.01f) continue;

            std::string badge = m.ModeBadge();
            std::string full  = badge.empty() ? m.name : (m.name + "  " + badge);
            ImVec2 ts = f->CalcTextSizeA(14.5f, FLT_MAX, 0.0f, full.c_str());
            active.push_back({ &m, badge, ts.x + 20.0f, vis });
        }

        std::sort(active.begin(), active.end(), [](const ArrayEntry& a, const ArrayEntry& b) {
            return a.width > b.width;
        });

        float curY = 16.0f;
        float rowH = 26.0f;
        for (size_t i = 0; i < active.size(); ++i) {
            const auto& e = active[i];
            float effH = rowH * e.animVis;
            float slideOffset = (1.0f - e.animVis) * (e.width + 24.0f);

            ImVec2 r0, r1;
            if (rightSide) {
                r1 = ImVec2(sw - 16.0f + slideOffset, curY + effH);
                r0 = ImVec2(r1.x - e.width, curY);
            } else {
                r0 = ImVec2(18.0f - slideOffset, curY);
                r1 = ImVec2(r0.x + e.width, curY + effH);
            }

            fgList->AddRectFilled(r0, r1, IM_COL32(14, 14, 18, static_cast<int>(222 * e.animVis)), 5.0f);

            float wave = 0.5f + 0.5f * std::sin(static_cast<float>(ImGui::GetTime()) * 2.4f - static_cast<float>(i) * 0.45f);
            ImVec4 barC(
                pal.accent.x + (pal.accentSecondary.x - pal.accent.x) * wave,
                pal.accent.y + (pal.accentSecondary.y - pal.accent.y) * wave,
                pal.accent.z + (pal.accentSecondary.z - pal.accent.z) * wave,
                e.animVis
            );
            if (rightSide) {
                fgList->AddRectFilled(ImVec2(r1.x - 3.0f, r0.y + 2.0f), ImVec2(r1.x, r1.y - 2.0f), ImGui::ColorConvertFloat4ToU32(barC), 1.5f);
            } else {
                fgList->AddRectFilled(ImVec2(r0.x, r0.y + 2.0f), ImVec2(r0.x + 3.0f, r1.y - 2.0f), ImGui::ColorConvertFloat4ToU32(barC), 1.5f);
            }

            ImVec2 ns = f->CalcTextSizeA(14.5f, FLT_MAX, 0.0f, e.mod->name.c_str());
            float tx = r0.x + 10.0f;
            float ty = r0.y + (effH - ns.y) * 0.5f;
            fgList->AddText(f, 14.5f, ImVec2(tx, ty), ImGui::ColorConvertFloat4ToU32(ImVec4(0.96f, 0.96f, 0.99f, e.animVis)), e.mod->name.c_str());
            if (!e.badge.empty()) {
                fgList->AddText(f, 14.5f, ImVec2(tx + ns.x + 6.0f, ty), ImGui::ColorConvertFloat4ToU32(ImVec4(0.62f, 0.64f, 0.72f, e.animVis)), e.badge.c_str());
            }
            curY += effH + 2.0f;
        }
    }

    // 3. Bottom-Right Slide-In Toasts
    bool showToasts = !mHud || (mHud->enabled && mHud->GetBool("toasts", true));
    float toastY = sh - 24.0f;
    for (int i = static_cast<int>(g_Toasts.size()) - 1; i >= 0; --i) {
        auto& t = g_Toasts[i];
        t.age += dt;
        if (t.age >= t.duration) {
            g_Toasts.erase(g_Toasts.begin() + i);
            continue;
        }
        if (!showToasts) continue;

        float targetSlide = (t.age > t.duration - 0.28f) ? 1.15f : 0.0f;
        t.slideX += (targetSlide - t.slideX) * std::min(1.0f, dt * 14.0f);

        float tw = 246.0f, th = 48.0f;
        float x0 = sw - tw - 20.0f + t.slideX * (tw + 32.0f);
        float y0 = toastY - th;
        ImVec2 p0(x0, y0), p1(x0 + tw, y0 + th);

        fgList->AddRectFilled(p0, p1, IM_COL32(16, 16, 20, 238), 9.0f);
        ImVec4 ac = t.positive ? pal.accent : ImVec4(0.55f, 0.56f, 0.62f, 1.0f);
        fgList->AddRect(p0, p1, ImGui::ColorConvertFloat4ToU32(ImVec4(ac.x, ac.y, ac.z, 0.42f)), 9.0f, 0, 1.1f);
        fgList->AddRectFilled(ImVec2(p0.x, p0.y + 8.0f), ImVec2(p0.x + 3.5f, p1.y - 8.0f), ImGui::ColorConvertFloat4ToU32(ac), 2.0f);

        ImFont* fb = g_Fonts.bold16 ? g_Fonts.bold16 : ImGui::GetFont();
        ImFont* fr = g_Fonts.reg14  ? g_Fonts.reg14  : ImGui::GetFont();
        fgList->AddText(fb, 13.8f, ImVec2(p0.x + 14.0f, p0.y + 7.5f), IM_COL32(246, 247, 252, 255), t.title.c_str());
        fgList->AddText(fr, 12.2f, ImVec2(p0.x + 14.0f, p0.y + 25.5f), IM_COL32(170, 174, 190, 245), t.detail.c_str());

        float prog = 1.0f - std::clamp(t.age / t.duration, 0.0f, 1.0f);
        fgList->AddRectFilled(ImVec2(p0.x + 8.0f, p1.y - 3.0f), ImVec2(p0.x + 8.0f + (tw - 16.0f) * prog, p1.y - 1.2f),
                              ImGui::ColorConvertFloat4ToU32(ac), 1.0f);

        toastY -= (th + 8.0f);
    }
}

// ============================================================================
// 10. CUSTOM SPRING-ANIMATED IMGUI WIDGETS & 1:1 CLOUDY 1.3 MENU SHELL
// ============================================================================
static void SetAnisoSamplerCallback(const ImDrawList*, const ImDrawCmd*) {
    // Custom sampler state applied around brand crest draws when needed
}

static void DrawRotatingChevron(ImDrawList* dl, ImVec2 center, float radius, float angleRad, ImU32 col, float thickness = 1.6f) {
    // Base points for a right-pointing chevron '>' rotated by angleRad (0 = right, pi/2 = down)
    ImVec2 pts[3] = {
        ImVec2(-radius * 0.45f, -radius * 0.75f),
        ImVec2( radius * 0.45f,  0.0f),
        ImVec2(-radius * 0.45f,  radius * 0.75f)
    };
    float c = std::cos(angleRad), s = std::sin(angleRad);
    for (auto& p : pts) {
        float nx = p.x * c - p.y * s;
        float ny = p.x * s + p.y * c;
        p = ImVec2(center.x + nx, center.y + ny);
    }
    dl->AddPolyline(pts, 3, col, ImDrawFlags_None, thickness);
}

static bool SpringToggleSwitch(const char* id, bool* v, const Palette& pal, float dt) {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems) return false;

    ImGuiID gid = window->GetID(id);
    float w = 36.0f, h = 19.0f;
    ImVec2 pos = window->DC.CursorPos;
    ImRect bb(pos, ImVec2(pos.x + w, pos.y + h));
    ImGui::ItemSize(bb);
    if (!ImGui::ItemAdd(bb, gid)) return false;

    bool hovered = false, held = false;
    bool pressed = ImGui::ButtonBehavior(bb, gid, &hovered, &held);
    if (pressed) {
        *v = !(*v);
        g_Sound.PlayToggle(*v);
    }

    float t = g_Anim.Step(gid, *v ? 1.0f : 0.0f, dt);
    float hov = g_Anim.Step(gid + 17, hovered ? 1.0f : 0.0f, dt);

    ImVec4 offBg = pal.bgWidget;
    ImVec4 onBg  = pal.accent;
    ImVec4 bg(
        offBg.x + (onBg.x - offBg.x) * t,
        offBg.y + (onBg.y - offBg.y) * t,
        offBg.z + (onBg.z - offBg.z) * t,
        1.0f
    );

    ImDrawList* dl = window->DrawList;
    if (t > 0.05f) {
        dl->AddRect(ImVec2(bb.Min.x - 1.5f, bb.Min.y - 1.5f), ImVec2(bb.Max.x + 1.5f, bb.Max.y + 1.5f),
                    ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, 0.22f * t)), h * 0.5f + 1.5f, 0, 1.5f);
    }
    dl->AddRectFilled(bb.Min, bb.Max, ImGui::ColorConvertFloat4ToU32(bg), h * 0.5f);
    dl->AddRect(bb.Min, bb.Max, ImGui::ColorConvertFloat4ToU32(ImVec4(1, 1, 1, 0.08f + 0.08f * hov)), h * 0.5f);

    float knobR = 6.8f + 0.6f * hov;
    float knobX = bb.Min.x + 9.5f + (w - 19.0f) * t;
    float knobY = (bb.Min.y + bb.Max.y) * 0.5f;
    dl->AddCircleFilled(ImVec2(knobX, knobY + 0.8f), knobR, IM_COL32(0, 0, 0, 75), 20);
    dl->AddCircleFilled(ImVec2(knobX, knobY), knobR, IM_COL32(252, 252, 255, 255), 20);
    return pressed;
}

static constexpr float kRowHToggle = 28.0f;
static constexpr float kRowHSlider = 38.0f;
static constexpr float kRowHCombo  = 32.0f;
static constexpr float kRowHColor  = 28.0f;

static void DrawTextEllipsis(ImDrawList* dl, ImFont* font, float fontSize, ImVec2 pos, float maxWidth, ImU32 col, const char* text) {
    if (!text || !text[0] || maxWidth <= 8.0f) return;
    ImFont* f = font ? font : ImGui::GetFont();
    ImVec2 fullSz = f->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, text);
    if (fullSz.x <= maxWidth) {
        dl->AddText(f, fontSize, pos, col, text);
        return;
    }
    const char* ell = "...";
    float ellW = f->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, ell).x;
    if (ellW >= maxWidth) return;

    std::string s(text);
    while (!s.empty()) {
        // Pop full UTF-8 codepoint
        while (!s.empty() && (static_cast<unsigned char>(s.back()) & 0xC0) == 0x80) s.pop_back();
        if (!s.empty()) s.pop_back();
        while (!s.empty() && s.back() == ' ') s.pop_back();
        std::string cand = s + ell;
        if (f->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, cand.c_str()).x <= maxWidth) {
            dl->AddText(f, fontSize, pos, col, cand.c_str());
            return;
        }
    }
}

static bool SpringSettingToggleRow(const char* id, const char* label, bool* v, float rowW, const Palette& pal, float dt) {
    ImGui::PushID(id);
    ImVec2 start = ImGui::GetCursorScreenPos();
    ImFont* f = g_Fonts.reg14 ? g_Fonts.reg14 : ImGui::GetFont();
    DrawTextEllipsis(ImGui::GetWindowDrawList(), f, 14.0f, ImVec2(start.x, start.y + 4.0f),
                     rowW - 46.0f, ImGui::ColorConvertFloat4ToU32(pal.textSecondary), label);
    ImGui::SetCursorScreenPos(ImVec2(start.x + rowW - 36.0f, start.y + 3.5f));
    bool changed = SpringToggleSwitch("##tgl", v, pal, dt);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    ImGui::SetCursorScreenPos(ImVec2(start.x, start.y + kRowHToggle));
    ImGui::Dummy(ImVec2(rowW, 0.0f));
    ImGui::PopStyleVar();
    ImGui::PopID();
    return changed;
}

static bool SpringSliderFloatRow(const char* id, const char* label, float* val, float vmin, float vmax,
                                 const char* fmt, float rowW, const Palette& pal, float dt) {
    ImGui::PushID(id);
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    ImVec2 pos   = ImGui::GetCursorScreenPos();
    ImFont* f    = g_Fonts.reg14 ? g_Fonts.reg14 : ImGui::GetFont();
    ImDrawList* dl = window->DrawList;

    char valBuf[48]{};
    std::snprintf(valBuf, sizeof(valBuf), fmt, *val);
    ImVec2 valSz = f->CalcTextSizeA(13.2f, FLT_MAX, 0.0f, valBuf);

    DrawTextEllipsis(dl, f, 13.8f, ImVec2(pos.x, pos.y + 1.0f),
                     rowW - valSz.x - 10.0f, ImGui::ColorConvertFloat4ToU32(pal.textSecondary), label);
    dl->AddText(f, 13.2f, ImVec2(pos.x + rowW - valSz.x, pos.y + 1.5f),
                ImGui::ColorConvertFloat4ToU32(pal.textPrimary), valBuf);

    // Inset bar ends by 6px so the circular thumb + drop shadow never clip against card edges at 0% or 100%
    const float knobPad = 6.0f;
    const float trackW  = std::max(12.0f, rowW - knobPad * 2.0f);
    ImVec2 barMin(pos.x + knobPad, pos.y + 22.0f);
    ImVec2 barMax(pos.x + rowW - knobPad, pos.y + 28.0f);
    ImRect hitBB(ImVec2(pos.x, barMin.y - 6.0f), ImVec2(pos.x + rowW, barMax.y + 6.0f));

    ImGuiID gid = window->GetID("##sld");
    ImGui::ItemSize(ImRect(pos, ImVec2(pos.x + rowW, pos.y + kRowHSlider)));
    if (!ImGui::ItemAdd(hitBB, gid)) { ImGui::PopID(); return false; }

    bool hovered = false, held = false;
    ImGui::ButtonBehavior(hitBB, gid, &hovered, &held);
    bool changed = false;
    if (held) {
        float mouseT = std::clamp((ImGui::GetIO().MousePos.x - barMin.x) / trackW, 0.0f, 1.0f);
        float newV = vmin + (vmax - vmin) * mouseT;
        if (std::fabs(newV - *val) > 1e-5f) {
            *val = newV;
            changed = true;
        }
    }

    float norm = std::clamp((*val - vmin) / std::max(1e-5f, vmax - vmin), 0.0f, 1.0f);
    float animNorm = g_Anim.Step(gid, norm, dt);
    float animHov  = g_Anim.Step(gid + 31, (hovered || held) ? 1.0f : 0.0f, dt);

    dl->AddRectFilled(barMin, barMax, ImGui::ColorConvertFloat4ToU32(pal.bgWidget), 3.0f);
    float fillX = barMin.x + trackW * animNorm;
    if (fillX > barMin.x + 1.0f) {
        dl->AddRectFilledMultiColor(
            barMin, ImVec2(fillX, barMax.y),
            ImGui::ColorConvertFloat4ToU32(pal.accent),
            ImGui::ColorConvertFloat4ToU32(pal.accentSecondary),
            ImGui::ColorConvertFloat4ToU32(pal.accentSecondary),
            ImGui::ColorConvertFloat4ToU32(pal.accent)
        );
    }
    float knobR = 5.2f + 1.2f * animHov;
    dl->AddCircleFilled(ImVec2(fillX, (barMin.y + barMax.y) * 0.5f), knobR + 1.2f, IM_COL32(0, 0, 0, 90), 20);
    dl->AddCircleFilled(ImVec2(fillX, (barMin.y + barMax.y) * 0.5f), knobR, IM_COL32(252, 252, 255, 255), 20);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y + kRowHSlider));
    ImGui::Dummy(ImVec2(rowW, 0.0f));
    ImGui::PopStyleVar();
    ImGui::PopID();
    return changed;
}

static bool SpringComboRow(const char* id, const char* label, int* currentItem,
                           const std::vector<std::string>& items, float rowW, const Palette& pal, float dt) {
    if (items.empty()) return false;
    *currentItem = std::clamp(*currentItem, 0, static_cast<int>(items.size()) - 1);

    ImGui::PushID(id);
    ImVec2 pos   = ImGui::GetCursorScreenPos();
    ImFont* f    = g_Fonts.reg14 ? g_Fonts.reg14 : ImGui::GetFont();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    float boxW = std::clamp(rowW * 0.48f, 108.0f, 146.0f);
    DrawTextEllipsis(dl, f, 13.8f, ImVec2(pos.x, pos.y + 5.0f),
                     rowW - boxW - 10.0f, ImGui::ColorConvertFloat4ToU32(pal.textSecondary), label);

    ImVec2 bMin(pos.x + rowW - boxW, pos.y + 2.0f);
    ImVec2 bMax(pos.x + rowW, pos.y + 27.0f);

    ImGui::SetCursorScreenPos(bMin);
    bool pressed = ImGui::InvisibleButton("##cmb_btn", ImVec2(boxW, 25.0f));
    bool hovered = ImGui::IsItemHovered();
    if (pressed) {
        ImGui::OpenPopup("##cmb_popup");
        g_Sound.PlayTab();
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
        *currentItem = (*currentItem + 1) % static_cast<int>(items.size());
        g_Sound.PlayTab();
    }

    bool popupOpen = ImGui::IsPopupOpen("##cmb_popup");
    float chevAng  = g_Anim.Step(ImGui::GetID("##cmb_ang"), popupOpen ? 1.570796f : 0.0f, dt);
    float hovA     = g_Anim.Step(ImGui::GetID("##cmb_hov"), (hovered || popupOpen) ? 1.0f : 0.0f, dt);

    ImVec4 bgCol(
        pal.bgWidget.x + (pal.bgWidgetHover.x - pal.bgWidget.x) * hovA,
        pal.bgWidget.y + (pal.bgWidgetHover.y - pal.bgWidget.y) * hovA,
        pal.bgWidget.z + (pal.bgWidgetHover.z - pal.bgWidget.z) * hovA,
        1.0f
    );
    dl->AddRectFilled(bMin, bMax, ImGui::ColorConvertFloat4ToU32(bgCol), 6.0f);
    dl->AddRect(bMin, bMax, ImGui::ColorConvertFloat4ToU32(popupOpen ? pal.accent : pal.border), 6.0f);

    const std::string& curStr = items[*currentItem];
    DrawTextEllipsis(dl, f, 13.0f, ImVec2(bMin.x + 9.0f, bMin.y + 4.5f),
                     boxW - 26.0f, ImGui::ColorConvertFloat4ToU32(pal.textPrimary), curStr.c_str());
    DrawRotatingChevron(dl, ImVec2(bMax.x - 11.0f, (bMin.y + bMax.y) * 0.5f), 4.2f, chevAng,
                        ImGui::ColorConvertFloat4ToU32(pal.textSecondary), 1.4f);

    bool changed = false;
    ImGui::SetNextWindowPos(ImVec2(bMin.x, bMax.y + 4.0f));
    ImGui::SetNextWindowSize(ImVec2(boxW, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 8.0f);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, pal.bgCard);
    ImGui::PushStyleColor(ImGuiCol_Border, pal.border);
    if (ImGui::BeginPopup("##cmb_popup")) {
        for (int i = 0; i < static_cast<int>(items.size()); ++i) {
            bool sel = (i == *currentItem);
            if (sel) ImGui::PushStyleColor(ImGuiCol_Text, pal.accent);
            else     ImGui::PushStyleColor(ImGuiCol_Text, pal.textPrimary);
            if (ImGui::Selectable(items[i].c_str(), sel, 0, ImVec2(0, 22.0f))) {
                *currentItem = i;
                changed = true;
                g_Sound.PlayTab();
            }
            ImGui::PopStyleColor();
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y + kRowHCombo));
    ImGui::Dummy(ImVec2(rowW, 0.0f));
    ImGui::PopStyleVar();
    ImGui::PopID();
    return changed;
}

static bool SpringColorRow(const char* id, const char* label, ImVec4* col, float rowW, const Palette& pal) {
    ImGui::PushID(id);
    ImVec2 pos   = ImGui::GetCursorScreenPos();
    ImFont* f    = g_Fonts.reg14 ? g_Fonts.reg14 : ImGui::GetFont();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    DrawTextEllipsis(dl, f, 13.8f, ImVec2(pos.x, pos.y + 4.0f),
                     rowW - 44.0f, ImGui::ColorConvertFloat4ToU32(pal.textSecondary), label);

    ImVec2 swMin(pos.x + rowW - 34.0f, pos.y + 3.0f);
    ImVec2 swMax(pos.x + rowW, pos.y + 22.0f);
    ImGui::SetCursorScreenPos(swMin);
    if (ImGui::InvisibleButton("##clr_btn", ImVec2(34.0f, 19.0f))) {
        ImGui::OpenPopup("##clr_picker");
    }
    dl->AddRectFilled(swMin, swMax, ImGui::ColorConvertFloat4ToU32(*col), 5.0f);
    dl->AddRect(swMin, swMax, ImGui::ColorConvertFloat4ToU32(pal.border), 5.0f, 0, 1.2f);

    bool changed = false;
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 10.0f));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, pal.bgCard);
    if (ImGui::BeginPopup("##clr_picker")) {
        changed = ImGui::ColorPicker4("##pkr", reinterpret_cast<float*>(col),
                                      ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_DisplayRGB);
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y + kRowHColor));
    ImGui::Dummy(ImVec2(rowW, 0.0f));
    ImGui::PopStyleVar();
    ImGui::PopID();
    return changed;
}

static bool s_KeyPrev[256]{};
static bool s_KeyTableInit   = false;
static bool s_WaitMouseClear = false;

static void ArmKeybindCapture(Module* targetMod, bool menuKey = false) {
    for (auto& m : g_Modules) {
        if (&m != targetMod) m.waitingKey = false;
    }
    if (!menuKey) g_Theme.rebindMenuKey = false;
    s_WaitMouseClear = true;
    for (int vk = 1; vk < 255; ++vk) {
        s_KeyPrev[vk] = (GetAsyncKeyState(vk) & 0x8000) != 0;
    }
    s_KeyTableInit = true;
}

static void PollModuleKeybinds(bool menuOpen) {
    if (!s_KeyTableInit) {
        for (int vk = 1; vk < 255; ++vk) {
            s_KeyPrev[vk] = (GetAsyncKeyState(vk) & 0x8000) != 0;
        }
        s_KeyTableInit = true;
    }

    bool keyDown[256]{};
    bool keyEdge[256]{};
    for (int vk = 1; vk < 255; ++vk) {
        bool d = (GetAsyncKeyState(vk) & 0x8000) != 0;
        keyDown[vk]   = d;
        keyEdge[vk]   = d && !s_KeyPrev[vk];
        s_KeyPrev[vk] = d;
    }

    // Handle menu toggle keybind capture
    if (g_Theme.rebindMenuKey) {
        if (s_WaitMouseClear) {
            if (!keyDown[VK_LBUTTON] && !keyDown[VK_RBUTTON] && !keyDown[VK_MBUTTON]) {
                s_WaitMouseClear = false;
            }
            return;
        }
        if (keyEdge[VK_ESCAPE] || keyEdge[VK_LBUTTON]) {
            g_Theme.rebindMenuKey = false;
            g_Sound.PlayToggle(false);
            return;
        }
        for (int vk = 8; vk < 254; ++vk) {
            if (!IsBindableVk(vk)) continue;
            if (keyEdge[vk]) {
                g_Theme.menuToggleVk  = vk;
                g_Theme.rebindMenuKey = false;
                g_Sound.PlayTab();
                PushToast("Menu Keybind", "Bound to " + VkToString(vk), true);
                return;
            }
        }
        return;
    }

    // Handle per-module keybind capture with strict whitelist & rising-edge guard
    for (auto& m : g_Modules) {
        if (!m.waitingKey) continue;
        if (s_WaitMouseClear) {
            if (!keyDown[VK_LBUTTON] && !keyDown[VK_RBUTTON] && !keyDown[VK_MBUTTON]) {
                s_WaitMouseClear = false;
            }
            return;
        }
        if (keyEdge[VK_ESCAPE] || keyEdge[VK_BACK]) {
            m.keybind    = 0;
            m.waitingKey = false;
            g_Sound.PlayToggle(false);
            PushToast(m.name, "Keybind cleared", false);
            return;
        }
        if (keyEdge[VK_LBUTTON]) {
            // Clicking anywhere with M1 cancels keybind capture cleanly
            m.waitingKey = false;
            g_Sound.PlayToggle(false);
            return;
        }
        for (int vk = 2; vk < 254; ++vk) {
            if (vk == g_Theme.menuToggleVk) continue;
            if (!IsBindableVk(vk)) continue;
            if (keyEdge[vk]) {
                m.keybind    = vk;
                m.waitingKey = false;
                m.keyWasDown = true;
                g_Sound.PlayTab();
                PushToast(m.name, "Bound to " + VkToString(vk), true);
                return;
            }
        }
        return;
    }

    if (menuOpen) return;

    // Runtime Toggle / Hold evaluation when playing
    for (auto& m : g_Modules) {
        if (m.keybind <= 0 || m.keybind >= 255) continue;

        // When Activation == "Hold Bind" (or Trigger Mode == "Hold Bind") and BindMode == Toggle,
        // keep the module enabled so holding its bound key activates tracking/firing without toggling it off.
        if (m.bindMode == BindMode::Toggle) {
            if (m.id == "aimbot" && m.GetCombo("activation", 0) == 4 && m.enabled) {
                m.keyWasDown = keyDown[m.keybind];
                continue;
            }
            if (m.id == "triggerbot" && m.GetCombo("mode", 0) == 1 && m.enabled) {
                m.keyWasDown = keyDown[m.keybind];
                continue;
            }
        }

        bool down = keyDown[m.keybind];
        if (m.bindMode == BindMode::Toggle) {
            if (down && !m.keyWasDown) {
                m.enabled = !m.enabled;
                g_Sound.PlayToggle(m.enabled);
                PushToast(m.name, m.enabled ? "Module enabled" : "Module disabled", m.enabled);
            }
        } else {
            if (down != m.enabled) {
                m.enabled = down;
                g_Sound.PlayToggle(m.enabled);
            }
        }
        m.keyWasDown = down;
    }
}

struct MenuNavState {
    int  activeTopTab      = 0; // 0=Combat, 1=Visuals, 2=Misc, 3=Configs, 4=Settings
    int  activeSubTab[5]   = { 0, 0, 0, 0, 0 };
    bool catExpanded[5]    = { true, true, false, false, false };
    char searchBuf[64]     = "";
    ImVec2 menuPos         = ImVec2(-1.0f, -1.0f);
    bool draggingMenu      = false;
    ImVec2 dragOffset      = ImVec2(0.0f, 0.0f);
};

static MenuNavState g_Nav;

static void RenderModuleCard(Module& m, float cardWidth, const Palette& pal, float dt) {
    ImGui::PushID(m.id.c_str());
    ImDrawList* dl    = ImGui::GetWindowDrawList();
    ImVec2 localStart = ImGui::GetCursorPos();
    ImVec2 cardStart  = ImGui::GetCursorScreenPos();

    // Exact pixel drawer height matching zero-ItemSpacing row steps
    float settingsTargetH = 0.0f;
    if (m.expanded && !m.settings.empty()) {
        settingsTargetH = 12.0f;
        for (const auto& s : m.settings) {
            switch (s.kind) {
                case SettingKind::Toggle:      settingsTargetH += kRowHToggle; break;
                case SettingKind::SliderFloat:
                case SettingKind::SliderInt:   settingsTargetH += kRowHSlider; break;
                case SettingKind::Combo:       settingsTargetH += kRowHCombo;  break;
                case SettingKind::Color:       settingsTargetH += kRowHColor;  break;
            }
        }
        settingsTargetH += 12.0f;
    }
    float animDrawerH = g_Anim.Step(ImGui::GetID("##drawer_h"), settingsTargetH, dt);
    float headerH     = 56.0f;
    float totalH      = headerH + animDrawerH;

    ImVec2 cardEnd(cardStart.x + cardWidth, cardStart.y + totalH);
    bool cardHovered = ImGui::IsMouseHoveringRect(cardStart, ImVec2(cardEnd.x, cardStart.y + headerH));
    float hovT = g_Anim.Step(ImGui::GetID("##card_hov"), cardHovered ? 1.0f : 0.0f, dt);
    float enT  = g_Anim.Step(ImGui::GetID("##card_en"),  m.enabled   ? 1.0f : 0.0f, dt);

    ImVec4 bg(
        pal.bgCard.x + (pal.bgCardHover.x - pal.bgCard.x) * hovT,
        pal.bgCard.y + (pal.bgCardHover.y - pal.bgCard.y) * hovT,
        pal.bgCard.z + (pal.bgCardHover.z - pal.bgCard.z) * hovT,
        pal.bgCard.w
    );
    dl->AddRectFilled(cardStart, cardEnd, ImGui::ColorConvertFloat4ToU32(bg), 11.0f);

    ImVec4 brd(
        pal.border.x + (pal.accent.x - pal.border.x) * (enT * 0.45f),
        pal.border.y + (pal.accent.y - pal.border.y) * (enT * 0.45f),
        pal.border.z + (pal.accent.z - pal.border.z) * (enT * 0.45f),
        pal.border.w + 0.18f * enT
    );
    dl->AddRect(cardStart, cardEnd, ImGui::ColorConvertFloat4ToU32(brd), 11.0f, 0, 1.1f);

    // Left accent bar when enabled
    if (enT > 0.02f) {
        dl->AddRectFilled(ImVec2(cardStart.x, cardStart.y + 12.0f),
                          ImVec2(cardStart.x + 3.0f, cardStart.y + headerH - 12.0f),
                          ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, enT)), 2.0f);
    }

    ImFont* fTitle = g_Fonts.bold16 ? g_Fonts.bold16 : ImGui::GetFont();
    ImFont* fDesc  = g_Fonts.reg14  ? g_Fonts.reg14  : ImGui::GetFont();

    // Right-side controls: [Keybind Pill] [Chevron] [Spring Toggle]
    float toggleX = cardEnd.x - 48.0f;
    float chevX   = m.settings.empty() ? (toggleX - 4.0f) : (toggleX - 24.0f);

    std::string bindStr = m.waitingKey ? "..." : VkToString(m.keybind);
    if (m.keybind > 0 && !m.waitingKey && m.bindMode == BindMode::Hold) {
        bindStr += " (H)";
    }
    ImVec2 bindTxtSz = fDesc->CalcTextSizeA(11.6f, FLT_MAX, 0.0f, bindStr.c_str());
    float bindW = std::max(38.0f, bindTxtSz.x + 14.0f);
    ImVec2 bindMin(chevX - bindW - 8.0f, cardStart.y + 18.0f);
    ImVec2 bindMax(chevX - 8.0f, cardStart.y + 38.0f);

    // Header title & cleanly ellipsized subtitle (never chopped mid-glyph)
    float maxTextW = std::max(40.0f, bindMin.x - (cardStart.x + 14.0f) - 8.0f);
    DrawTextEllipsis(dl, fTitle, 15.2f, ImVec2(cardStart.x + 14.0f, cardStart.y + 9.5f),
                     maxTextW, ImGui::ColorConvertFloat4ToU32(pal.textPrimary), m.name.c_str());
    DrawTextEllipsis(dl, fDesc, 12.3f, ImVec2(cardStart.x + 14.0f, cardStart.y + 30.5f),
                     maxTextW, ImGui::ColorConvertFloat4ToU32(pal.textMuted), m.desc.c_str());

    ImGui::SetCursorScreenPos(bindMin);
    if (ImGui::InvisibleButton("##bind_btn", ImVec2(bindW, 20.0f))) {
        bool nextWait = !m.waitingKey;
        m.waitingKey  = nextWait;
        if (nextWait) ArmKeybindCapture(&m, false);
        g_Sound.PlayTab();
    }
    bool bindHov = ImGui::IsItemHovered();
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
        m.waitingKey = false;
        ImGui::OpenPopup("##bind_mode_pop");
    }

    ImVec4 pillBg = m.waitingKey ? pal.accentGlow : (bindHov ? pal.bgWidgetHover : pal.bgWidget);
    ImVec4 pillBd = m.waitingKey ? pal.accent     : (m.keybind > 0 ? pal.border : pal.borderSubtle);
    ImVec4 pillFg = m.waitingKey ? pal.accent     : (m.keybind > 0 ? pal.textPrimary : pal.textSecondary);
    dl->AddRectFilled(bindMin, bindMax, ImGui::ColorConvertFloat4ToU32(pillBg), 5.0f);
    dl->AddRect(bindMin, bindMax, ImGui::ColorConvertFloat4ToU32(pillBd), 5.0f);
    dl->AddText(fDesc, 11.6f, ImVec2(bindMin.x + (bindW - bindTxtSz.x) * 0.5f, bindMin.y + 3.0f),
                ImGui::ColorConvertFloat4ToU32(pillFg), bindStr.c_str());

    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, pal.bgCard);
    ImGui::PushStyleColor(ImGuiCol_Border, pal.border);
    if (ImGui::BeginPopup("##bind_mode_pop")) {
        if (ImGui::Selectable("Mode: Toggle", m.bindMode == BindMode::Toggle)) m.bindMode = BindMode::Toggle;
        if (ImGui::Selectable("Mode: Hold",   m.bindMode == BindMode::Hold))   m.bindMode = BindMode::Hold;
        if (ImGui::Selectable("Clear Keybind", false)) m.keybind = 0;
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);

    // Settings Chevron Button
    if (!m.settings.empty()) {
        ImGui::SetCursorScreenPos(ImVec2(chevX - 4.0f, cardStart.y + 16.0f));
        if (ImGui::InvisibleButton("##exp_btn", ImVec2(22.0f, 24.0f))) {
            m.expanded = !m.expanded;
            g_Sound.PlayTab();
        }
        float cAng = g_Anim.Step(ImGui::GetID("##exp_ang"), m.expanded ? 1.570796f : 0.0f, dt);
        DrawRotatingChevron(dl, ImVec2(chevX + 7.0f, cardStart.y + 28.0f), 4.6f, cAng,
                            ImGui::ColorConvertFloat4ToU32(m.expanded ? pal.accent : pal.textSecondary), 1.6f);
    }

    // Header click on left region toggles module; right-click expands settings; middle-click arms keybind
    ImGui::SetCursorScreenPos(cardStart);
    if (ImGui::InvisibleButton("##hdr_hit", ImVec2(std::max(24.0f, bindMin.x - cardStart.x - 4.0f), headerH))) {
        m.enabled = !m.enabled;
        g_Sound.PlayToggle(m.enabled);
        PushToast(m.name, m.enabled ? "Module enabled" : "Module disabled", m.enabled);
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right) && !m.settings.empty()) {
        m.expanded = !m.expanded;
        g_Sound.PlayTab();
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Middle)) {
        bool nextWait = !m.waitingKey;
        m.waitingKey  = nextWait;
        if (nextWait) ArmKeybindCapture(&m, false);
        g_Sound.PlayTab();
    }

    // Enable Switch
    ImGui::SetCursorScreenPos(ImVec2(toggleX, cardStart.y + 18.5f));
    bool prevEn = m.enabled;
    if (SpringToggleSwitch("##mod_sw", &m.enabled, pal, dt)) {
        if (m.enabled != prevEn) {
            PushToast(m.name, m.enabled ? "Module enabled" : "Module disabled", m.enabled);
        }
    }

    // Animated Settings Drawer with explicit inner width (zero WorkRect hacks, zero bottom clipping)
    if (animDrawerH > 1.5f && !m.settings.empty()) {
        dl->PushClipRect(ImVec2(cardStart.x + 1.0f, cardStart.y + headerH),
                         ImVec2(cardEnd.x - 1.0f, cardEnd.y - 1.0f), true);
        dl->AddLine(ImVec2(cardStart.x + 14.0f, cardStart.y + headerH),
                    ImVec2(cardEnd.x - 14.0f, cardStart.y + headerH),
                    ImGui::ColorConvertFloat4ToU32(pal.borderSubtle), 1.0f);

        const float innerW = cardWidth - 28.0f;
        ImGui::SetCursorScreenPos(ImVec2(cardStart.x + 14.0f, cardStart.y + headerH + 12.0f));
        for (auto& s : m.settings) {
            ImGui::PushID(s.id.c_str());
            switch (s.kind) {
                case SettingKind::Toggle:
                    SpringSettingToggleRow(s.id.c_str(), s.label.c_str(), &s.bVal, innerW, pal, dt);
                    break;
                case SettingKind::SliderFloat:
                    SpringSliderFloatRow(s.id.c_str(), s.label.c_str(), &s.fVal, s.fMin, s.fMax, s.fmt.c_str(), innerW, pal, dt);
                    break;
                case SettingKind::Combo:
                    SpringComboRow(s.id.c_str(), s.label.c_str(), &s.iVal, s.items, innerW, pal, dt);
                    break;
                case SettingKind::Color:
                    SpringColorRow(s.id.c_str(), s.label.c_str(), &s.cVal, innerW, pal);
                    break;
                default:
                    break;
            }
            ImGui::PopID();
        }
        dl->PopClipRect();
    }

    // Advance window-local cursor by exact card height + 12px vertical gap
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    ImGui::SetCursorPos(ImVec2(localStart.x, localStart.y + totalH + 12.0f));
    ImGui::Dummy(ImVec2(cardWidth, 0.0f));
    ImGui::PopStyleVar();
    ImGui::PopID();
}

static void RenderRelinquishMenu(const WorldSnapshot& world, const Palette& pal, float sw, float sh, float dt,
                                 float menuAlpha, bool& requestExit, ImVec2& outMin, ImVec2& outMax) {
    const float menuW = std::min(900.0f, std::max(680.0f, sw - 28.0f));
    const float menuH = std::min(585.0f, std::max(460.0f, sh - 28.0f));
    if (g_Nav.menuPos.x < 0.0f) {
        g_Nav.menuPos = ImVec2(std::floor((sw - menuW) * 0.5f), std::floor((sh - menuH) * 0.5f));
    }

    // Smooth scale-in pop on open
    float popScale = 0.96f + 0.04f * menuAlpha;
    ImVec2 center(g_Nav.menuPos.x + menuW * 0.5f, g_Nav.menuPos.y + menuH * 0.5f);
    outMin = ImVec2(center.x - (menuW * popScale) * 0.5f, center.y - (menuH * popScale) * 0.5f);
    outMax = ImVec2(center.x + (menuW * popScale) * 0.5f, center.y + (menuH * popScale) * 0.5f);

    ImGui::SetNextWindowPos(g_Nav.menuPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(menuW, menuH), ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, menuAlpha);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 16.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove;

    if (ImGui::Begin("##RelinquishMainShell", nullptr, flags)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 wMin = ImGui::GetWindowPos();
        ImVec2 wMax(wMin.x + menuW, wMin.y + menuH);
        const float sideW = 220.0f;
        const float topH  = 58.0f;

        // Render HLSL backdrop directly inside this WindowDrawList (AFTER BackgroundDrawList!)
        // so FOV circles, boxes, snaplines & crosshairs behind the menu are 100% occluded.
        if (g_Shaders.backdropEnabled && g_Shaders.ready) {
            ShaderParamsCB& cb = g_Shaders.pendingBackdrop;
            cb = {};
            cb.rectMin[0]    = wMin.x;
            cb.rectMin[1]    = wMin.y;
            cb.rectMax[0]    = wMax.x;
            cb.rectMax[1]    = wMax.y;
            cb.screenSize[0] = sw;
            cb.screenSize[1] = sh;
            cb.time          = static_cast<float>(ImGui::GetTime());
            cb.alpha         = menuAlpha;
            std::memcpy(cb.accentA, &pal.accent,          sizeof(float) * 4);
            std::memcpy(cb.accentB, &pal.accentSecondary, sizeof(float) * 4);
            std::memcpy(cb.bgBase,  &pal.bgWindow,        sizeof(float) * 4);
            cb.rounding      = 16.0f;
            cb.sheenStrength = g_Shaders.sheenIntensity;
            cb.glowStrength  = g_Shaders.glowIntensity;

            dl->AddCallback(MenuShaderRenderer::BackdropDrawCallback, &g_Shaders);
            dl->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
        } else {
            dl->AddRectFilled(wMin, wMax, ImGui::ColorConvertFloat4ToU32(
                              ImVec4(pal.bgWindow.x, pal.bgWindow.y, pal.bgWindow.z, menuAlpha)), 16.0f);
        }

        // Sidebar panel + dividers + outer border
        dl->AddRectFilled(wMin, ImVec2(wMin.x + sideW, wMax.y), ImGui::ColorConvertFloat4ToU32(pal.bgSidebar),
                          16.0f, ImDrawFlags_RoundCornersLeft);
        dl->AddLine(ImVec2(wMin.x + sideW, wMin.y + 1.0f), ImVec2(wMin.x + sideW, wMax.y - 1.0f),
                    ImGui::ColorConvertFloat4ToU32(pal.border), 1.0f);
        dl->AddLine(ImVec2(wMin.x + sideW, wMin.y + topH), ImVec2(wMax.x, wMin.y + topH),
                    ImGui::ColorConvertFloat4ToU32(pal.borderSubtle), 1.0f);
        dl->AddRect(wMin, wMax, ImGui::ColorConvertFloat4ToU32(pal.border), 16.0f, 0, 1.2f);

        // ---------------- LEFT SIDEBAR: FIXED CREST + SCROLLABLE NAV + FIXED STATUS FOOTER ----------------
        float navTopY  = wMin.y + 84.0f;
        float footTopY = wMax.y - 54.0f;

        if (g_BrandLogo.srv) {
            float lw = 156.0f;
            float lh = lw * (static_cast<float>(g_BrandLogo.height) / std::max(1.0f, static_cast<float>(g_BrandLogo.width)));
            lh = std::min(lh, 82.0f);
            ImVec2 lp(wMin.x + (sideW - lw) * 0.5f, wMin.y + 8.0f);
            dl->AddImage(reinterpret_cast<ImTextureID>(g_BrandLogo.srv), lp, ImVec2(lp.x + lw, lp.y + lh));
            navTopY = lp.y + lh + 6.0f;
        } else {
            ImFont* fb = g_Fonts.bold20 ? g_Fonts.bold20 : ImGui::GetFont();
            ImVec2 ts = fb->CalcTextSizeA(20.0f, FLT_MAX, 0.0f, "RELINQUISH");
            dl->AddText(fb, 20.0f, ImVec2(wMin.x + (sideW - ts.x) * 0.5f, wMin.y + 26.0f),
                        ImGui::ColorConvertFloat4ToU32(pal.textPrimary), "RELINQUISH");
            navTopY = wMin.y + 72.0f;
        }
        dl->AddLine(ImVec2(wMin.x + 16.0f, navTopY), ImVec2(wMin.x + sideW - 16.0f, navTopY),
                    ImGui::ColorConvertFloat4ToU32(pal.borderSubtle), 1.0f);
        navTopY += 6.0f;

        // Drag handle across top bar (excluding search box) & sidebar brand header
        ImRect dragZoneA(wMin, ImVec2(wMin.x + sideW, navTopY - 4.0f));
        ImRect dragZoneB(ImVec2(wMin.x + sideW, wMin.y), ImVec2(wMax.x - 245.0f, wMin.y + topH));
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
            (dragZoneA.Contains(ImGui::GetIO().MousePos) || dragZoneB.Contains(ImGui::GetIO().MousePos))) {
            g_Nav.draggingMenu = true;
            g_Nav.dragOffset   = ImVec2(ImGui::GetIO().MousePos.x - g_Nav.menuPos.x, ImGui::GetIO().MousePos.y - g_Nav.menuPos.y);
        }
        if (g_Nav.draggingMenu) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                g_Nav.menuPos.x = std::clamp(ImGui::GetIO().MousePos.x - g_Nav.dragOffset.x, 4.0f, std::max(4.0f, sw - menuW - 4.0f));
                g_Nav.menuPos.y = std::clamp(ImGui::GetIO().MousePos.y - g_Nav.dragOffset.y, 4.0f, std::max(4.0f, sh - menuH - 4.0f));
            } else {
                g_Nav.draggingMenu = false;
            }
        }

        // Scrollable Sidebar Category Tree (strictly bounded above the status pill!)
        ImGui::SetCursorScreenPos(ImVec2(wMin.x, navTopY));
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 4.0f);
        ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, ImVec4(0.55f, 0.56f, 0.62f, 0.28f));
        ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabHovered, pal.accent);
        ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabActive, pal.accent);
        ImGui::BeginChild("##SidebarNav", ImVec2(sideW, std::max(60.0f, footTopY - navTopY)), false, ImGuiWindowFlags_NoBackground);
        {
            ImDrawList* sdl = ImGui::GetWindowDrawList();
            struct CatDef {
                const char* title;
                const char* icon;
                std::vector<const char*> subs;
            };
            static const CatDef kCats[5] = {
                { "Combat",   ICON_FA_CROSSHAIRS, { "Aim", "Trigger" } },
                { "Visuals",  ICON_FA_EYE,        { "Players", "World", "HUD" } },
                { "Misc",     ICON_FA_SLIDERS,    { "Utility" } },
                { "Configs",  ICON_FA_FOLDER,     { "Profiles", "Categories", "Client" } },
                { "Settings", ICON_FA_GEAR,       { "Menu", "Colour", "Shaders", "Animation", "Audio" } }
            };

            ImFont* fCat = g_Fonts.med15 ? g_Fonts.med15 : ImGui::GetFont();
            ImFont* fSub = g_Fonts.reg14 ? g_Fonts.reg14 : ImGui::GetFont();

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
            ImGui::Dummy(ImVec2(sideW - 8.0f, 2.0f));

            for (int ci = 0; ci < 5; ++ci) {
                if (ci == 3) {
                    ImVec2 sepP = ImGui::GetCursorScreenPos();
                    sdl->AddLine(ImVec2(sepP.x + 18.0f, sepP.y + 6.0f), ImVec2(sepP.x + sideW - 18.0f, sepP.y + 6.0f),
                                 ImGui::ColorConvertFloat4ToU32(pal.borderSubtle), 1.0f);
                    ImGui::Dummy(ImVec2(sideW - 8.0f, 13.0f));
                }

                ImGui::PushID(ci);
                ImVec2 rowLocal = ImGui::GetCursorPos();
                ImVec2 rowP     = ImGui::GetCursorScreenPos();
                ImVec2 rMin(rowP.x + 12.0f, rowP.y);
                ImVec2 rMax(rowP.x + sideW - 14.0f, rowP.y + 34.0f);

                ImGui::SetCursorScreenPos(rMin);
                if (ImGui::InvisibleButton("##cat_btn", ImVec2(rMax.x - rMin.x, 34.0f))) {
                    if (g_Nav.activeTopTab == ci) {
                        g_Nav.catExpanded[ci] = !g_Nav.catExpanded[ci];
                    } else {
                        g_Nav.activeTopTab    = ci;
                        g_Nav.catExpanded[ci] = true;
                        g_Nav.searchBuf[0]    = '\0';
                    }
                    g_Sound.PlayTab();
                }

                bool isSel = (g_Nav.activeTopTab == ci);
                bool isHov = ImGui::IsItemHovered();
                float selA = g_Anim.Step(ImGui::GetID("##csel"), isSel ? 1.0f : (isHov ? 0.35f : 0.0f), dt);
                if (selA > 0.02f) {
                    sdl->AddRectFilled(rMin, rMax, ImGui::ColorConvertFloat4ToU32(ImVec4(pal.bgWidget.x, pal.bgWidget.y, pal.bgWidget.z, selA)), 8.0f);
                }
                if (isSel) {
                    sdl->AddRectFilled(ImVec2(rMin.x, rMin.y + 8.0f), ImVec2(rMin.x + 3.0f, rMax.y - 8.0f),
                                       ImGui::ColorConvertFloat4ToU32(pal.accent), 2.0f);
                }

                ImU32 iconCol = ImGui::ColorConvertFloat4ToU32(isSel ? pal.accent : pal.textSecondary);
                ImU32 txtCol  = ImGui::ColorConvertFloat4ToU32(isSel ? pal.textPrimary : pal.textSecondary);
                sdl->AddText(fCat, 15.0f, ImVec2(rMin.x + 12.0f, rMin.y + 8.0f), iconCol, kCats[ci].icon);
                sdl->AddText(fCat, 15.0f, ImVec2(rMin.x + 36.0f, rMin.y + 8.0f), txtCol, kCats[ci].title);

                float chevA = g_Anim.Step(ImGui::GetID("##cchev"), g_Nav.catExpanded[ci] ? 1.570796f : 0.0f, dt);
                DrawRotatingChevron(sdl, ImVec2(rMax.x - 12.0f, (rMin.y + rMax.y) * 0.5f), 4.0f, chevA,
                                    ImGui::ColorConvertFloat4ToU32(pal.textMuted), 1.4f);

                // Animated Subtab Tree Drawer
                float targetSubH = g_Nav.catExpanded[ci] ? (static_cast<float>(kCats[ci].subs.size()) * 27.0f + 4.0f) : 0.0f;
                float animSubH   = g_Anim.Step(ImGui::GetID("##csubh"), targetSubH, dt);
                ImVec2 subBase(rowP.x, rMax.y + 2.0f);

                if (animSubH > 1.0f) {
                    ImVec2 clip0(subBase.x, std::max(navTopY, subBase.y));
                    ImVec2 clip1(subBase.x + sideW, std::min(footTopY, subBase.y + animSubH));
                    if (clip1.y > clip0.y) {
                        sdl->PushClipRect(clip0, clip1, true);
                        sdl->AddLine(ImVec2(rowP.x + 30.0f, subBase.y + 3.0f),
                                     ImVec2(rowP.x + 30.0f, subBase.y + animSubH - 5.0f),
                                     ImGui::ColorConvertFloat4ToU32(pal.border), 1.0f);

                        for (int si = 0; si < static_cast<int>(kCats[ci].subs.size()); ++si) {
                            ImGui::PushID(si);
                            ImVec2 sMin(rowP.x + 38.0f, subBase.y + si * 27.0f + 1.0f);
                            ImVec2 sMax(rowP.x + sideW - 16.0f, sMin.y + 24.0f);
                            ImGui::SetCursorScreenPos(sMin);
                            if (ImGui::InvisibleButton("##sub_btn", ImVec2(sMax.x - sMin.x, 24.0f))) {
                                g_Nav.activeTopTab     = ci;
                                g_Nav.activeSubTab[ci] = si;
                                g_Nav.searchBuf[0]     = '\0';
                                g_Sound.PlayTab();
                            }
                            bool subActive = (g_Nav.activeTopTab == ci && g_Nav.activeSubTab[ci] == si);
                            bool subHov    = ImGui::IsItemHovered();
                            if (subActive) {
                                sdl->AddCircleFilled(ImVec2(rowP.x + 30.0f, (sMin.y + sMax.y) * 0.5f), 2.8f,
                                                     ImGui::ColorConvertFloat4ToU32(pal.accent), 12);
                                sdl->AddRectFilled(sMin, sMax, ImGui::ColorConvertFloat4ToU32(
                                                   ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, 0.14f)), 6.0f);
                            }
                            sdl->AddText(fSub, 13.6f, ImVec2(sMin.x + 10.0f, sMin.y + 3.5f),
                                         ImGui::ColorConvertFloat4ToU32(subActive ? pal.accent : (subHov ? pal.textPrimary : pal.textMuted)),
                                         kCats[ci].subs[si]);
                            ImGui::PopID();
                        }
                        sdl->PopClipRect();
                    }
                }

                ImGui::SetCursorPos(ImVec2(rowLocal.x, rowLocal.y + 36.0f + animSubH + 2.0f));
                ImGui::Dummy(ImVec2(sideW - 8.0f, 0.0f));
                ImGui::PopID();
            }

            ImGui::Dummy(ImVec2(sideW - 8.0f, 12.0f));
            ImGui::PopStyleVar();
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar();

        // Fixed Bottom Process Status Pill inside Sidebar Footer (never overlapped by tree!)
        dl->AddLine(ImVec2(wMin.x + 16.0f, footTopY), ImVec2(wMin.x + sideW - 16.0f, footTopY),
                    ImGui::ColorConvertFloat4ToU32(pal.borderSubtle), 1.0f);
        ImVec2 footMin(wMin.x + 14.0f, footTopY + 10.0f);
        ImVec2 footMax(wMin.x + sideW - 14.0f, wMax.y - 12.0f);
        dl->AddRectFilled(footMin, footMax, ImGui::ColorConvertFloat4ToU32(pal.bgCard), 8.0f);
        dl->AddRect(footMin, footMax, ImGui::ColorConvertFloat4ToU32(pal.borderSubtle), 8.0f);
        ImU32 dotCol = world.attached ? IM_COL32(72, 232, 148, 255) : IM_COL32(250, 178, 72, 255);
        dl->AddCircleFilled(ImVec2(footMin.x + 13.0f, (footMin.y + footMax.y) * 0.5f), 4.0f, dotCol, 16);
        char stTxt[48]{};
        if (world.attached) std::snprintf(stTxt, sizeof(stTxt), "Attached (%lu)", static_cast<unsigned long>(world.pid));
        else                std::snprintf(stTxt, sizeof(stTxt), "Awaiting Game...");
        ImFont* fFoot = g_Fonts.reg14 ? g_Fonts.reg14 : ImGui::GetFont();
        dl->AddText(fFoot, 12.6f, ImVec2(footMin.x + 25.0f, (footMin.y + footMax.y) * 0.5f - 6.5f),
                    ImGui::ColorConvertFloat4ToU32(pal.textSecondary), stTxt);

        // ---------------- TOP HEADER BAR (BREADCRUMB + ANIMATED SEARCH) ----------------
        static const char* kCatNames[5] = { "Combat", "Visuals", "Misc", "Configs", "Settings" };
        static const std::vector<const char*> kSubNames[5] = {
            { "Aim", "Trigger" },
            { "Players", "World", "HUD" },
            { "Utility" },
            { "Profiles", "Categories", "Client" },
            { "Menu", "Colour", "Shaders", "Animation", "Audio" }
        };

        int topIdx = std::clamp(g_Nav.activeTopTab, 0, 4);
        int subIdx = std::clamp(g_Nav.activeSubTab[topIdx], 0, static_cast<int>(kSubNames[topIdx].size()) - 1);

        ImFont* fHead = g_Fonts.bold16 ? g_Fonts.bold16 : ImGui::GetFont();
        ImFont* fReg  = g_Fonts.reg14  ? g_Fonts.reg14  : ImGui::GetFont();
        bool hasSearch = (g_Nav.searchBuf[0] != '\0');

        float crumbX = wMin.x + sideW + 22.0f;
        float crumbY = wMin.y + 20.0f;
        dl->AddCircleFilled(ImVec2(crumbX, crumbY + 8.0f), 3.5f, ImGui::ColorConvertFloat4ToU32(pal.accent), 12);
        if (hasSearch) {
            dl->AddText(fHead, 16.0f, ImVec2(crumbX + 12.0f, crumbY - 1.0f), ImGui::ColorConvertFloat4ToU32(pal.textPrimary), "Search Results");
        } else {
            std::string c1 = kCatNames[topIdx];
            ImVec2 c1Sz = fHead->CalcTextSizeA(16.0f, FLT_MAX, 0.0f, c1.c_str());
            dl->AddText(fHead, 16.0f, ImVec2(crumbX + 12.0f, crumbY - 1.0f), ImGui::ColorConvertFloat4ToU32(pal.textPrimary), c1.c_str());
            DrawRotatingChevron(dl, ImVec2(crumbX + 22.0f + c1Sz.x, crumbY + 8.0f), 3.8f, 0.0f, ImGui::ColorConvertFloat4ToU32(pal.textMuted), 1.4f);
            dl->AddText(fHead, 16.0f, ImVec2(crumbX + 33.0f + c1Sz.x, crumbY - 1.0f), ImGui::ColorConvertFloat4ToU32(pal.textSecondary), kSubNames[topIdx][subIdx]);
        }

        // Animated Search Pill
        float searchW = 220.0f;
        ImVec2 sMin(wMax.x - searchW - 20.0f, wMin.y + 13.0f);
        ImVec2 sMax(wMax.x - 20.0f, wMin.y + 45.0f);
        dl->AddRectFilled(sMin, sMax, ImGui::ColorConvertFloat4ToU32(pal.bgCard), 9.0f);
        dl->AddRect(sMin, sMax, ImGui::ColorConvertFloat4ToU32(hasSearch ? pal.accent : pal.border), 9.0f, 0, 1.1f);
        dl->AddText(fReg, 14.0f, ImVec2(sMin.x + 11.0f, sMin.y + 8.0f), ImGui::ColorConvertFloat4ToU32(pal.textMuted), ICON_FA_MAGNIFYING);

        ImGui::SetCursorScreenPos(ImVec2(sMin.x + 31.0f, sMin.y + 6.5f));
        ImGui::PushItemWidth(searchW - (hasSearch ? 56.0f : 38.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_Text, pal.textPrimary);
        ImGui::InputTextWithHint("##GlobalSearch", "Search modules...", g_Nav.searchBuf, sizeof(g_Nav.searchBuf));
        ImGui::PopStyleColor(2);
        ImGui::PopItemWidth();

        if (hasSearch) {
            ImGui::SetCursorScreenPos(ImVec2(sMax.x - 24.0f, sMin.y + 6.0f));
            if (ImGui::InvisibleButton("##clr_srch", ImVec2(20.0f, 20.0f))) {
                g_Nav.searchBuf[0] = '\0';
            }
            dl->AddText(fReg, 13.0f, ImVec2(sMax.x - 19.0f, sMin.y + 8.5f), ImGui::ColorConvertFloat4ToU32(pal.textSecondary), ICON_FA_XMARK);
        }

        // ---------------- MAIN CONTENT SCROLL CANVAS ----------------
        ImGui::SetCursorScreenPos(ImVec2(wMin.x + sideW, wMin.y + topH));
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 6.0f);
        ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, ImVec4(0.55f, 0.56f, 0.62f, 0.30f));
        ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabHovered, pal.accent);
        ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabActive, pal.accent);
        ImGui::BeginChild("##MainContent", ImVec2(menuW - sideW, menuH - topH), false, ImGuiWindowFlags_NoBackground);
        {
            const float padX     = 20.0f;
            const float padTop   = 16.0f;
            const float contentW = (menuW - sideW) - padX * 2.0f - 4.0f;
            ImGui::SetCursorPos(ImVec2(padX, padTop));

            if (hasSearch || topIdx <= 2) {
                // Two-column balanced module masonry grid using window-local Y so ScrollMax.y is 100% exact
                std::string q = g_Nav.searchBuf;
                std::transform(q.begin(), q.end(), q.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

                std::vector<Module*> filtered;
                for (auto& m : g_Modules) {
                    if (hasSearch) {
                        std::string n = m.name + " " + m.desc;
                        std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                        if (n.find(q) == std::string::npos) continue;
                    } else {
                        if (m.category != topIdx || m.subtab != subIdx) continue;
                    }
                    filtered.push_back(&m);
                }

                float colGap = 14.0f;
                float colW   = std::floor((contentW - colGap) * 0.5f);

                if (filtered.empty()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, pal.textMuted);
                    ImGui::TextUnformatted("No matching modules found.");
                    ImGui::PopStyleColor();
                    ImGui::Dummy(ImVec2(contentW, 16.0f));
                } else {
                    ImGui::SetCursorPos(ImVec2(padX, padTop));
                    for (size_t i = 0; i < filtered.size(); i += 2) {
                        RenderModuleCard(*filtered[i], colW, pal, dt);
                    }
                    float leftLocalEnd = ImGui::GetCursorPosY();

                    float rightLocalEnd = padTop;
                    if (filtered.size() > 1) {
                        ImGui::SetCursorPos(ImVec2(padX + colW + colGap, padTop));
                        for (size_t i = 1; i < filtered.size(); i += 2) {
                            RenderModuleCard(*filtered[i], colW, pal, dt);
                        }
                        rightLocalEnd = ImGui::GetCursorPosY();
                    }

                    ImGui::SetCursorPos(ImVec2(padX, std::max(leftLocalEnd, rightLocalEnd)));
                    ImGui::Dummy(ImVec2(contentW, 24.0f));
                }
            } else if (topIdx == 3) {
                // ---------------- CONFIGS PAGE ----------------
                ImDrawList* cdl = ImGui::GetWindowDrawList();

                if (subIdx == 0) {
                    ImVec2 card0 = ImGui::GetCursorScreenPos();
                    ImVec2 card1(card0.x + contentW, card0.y + 62.0f);
                    cdl->AddRectFilled(card0, card1, ImGui::ColorConvertFloat4ToU32(pal.bgCard), 11.0f);
                    cdl->AddRect(card0, card1, ImGui::ColorConvertFloat4ToU32(pal.border), 11.0f);

                    ImGui::SetCursorScreenPos(ImVec2(card0.x + 14.0f, card0.y + 16.0f));
                    ImGui::PushItemWidth(250.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 6.5f));
                    ImGui::PushStyleColor(ImGuiCol_FrameBg, pal.bgWidget);
                    ImGui::PushStyleColor(ImGuiCol_Text, pal.textPrimary);
                    ImGui::InputTextWithHint("##NewCfgName", "Profile name...", g_Configs.newProfileBuf, sizeof(g_Configs.newProfileBuf));
                    ImGui::PopStyleColor(2);
                    ImGui::PopStyleVar(2);
                    ImGui::PopItemWidth();

                    ImGui::SameLine(0.0f, 10.0f);
                    ImGui::PushStyleColor(ImGuiCol_Button, pal.accent);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, pal.accentSecondary);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7.0f);
                    if (ImGui::Button(ICON_FA_PLUS "  Create & Save", ImVec2(142.0f, 30.0f))) {
                        if (g_Configs.Save(g_Configs.newProfileBuf)) {
                            g_Sound.PlayToggle(true);
                            PushToast("Profile Saved", g_Configs.newProfileBuf, true);
                        }
                    }
                    ImGui::PopStyleVar();
                    ImGui::PopStyleColor(3);

                    ImGui::SameLine(0.0f, 10.0f);
                    ImGui::PushStyleColor(ImGuiCol_Button, pal.bgWidget);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, pal.bgWidgetHover);
                    ImGui::PushStyleColor(ImGuiCol_Text, pal.textPrimary);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7.0f);
                    if (ImGui::Button(ICON_FA_FOLDER "  Refresh", ImVec2(106.0f, 30.0f))) {
                        g_Configs.RefreshList();
                        g_Sound.PlayTab();
                    }
                    ImGui::PopStyleVar();
                    ImGui::PopStyleColor(3);

                    ImGui::SetCursorPos(ImVec2(padX, padTop + 76.0f));
                    for (size_t i = 0; i < g_Configs.profiles.size(); ++i) {
                        ImGui::PushID(static_cast<int>(i));
                        const std::string& pname = g_Configs.profiles[i];
                        ImVec2 rowLocal = ImGui::GetCursorPos();
                        ImVec2 r0       = ImGui::GetCursorScreenPos();
                        ImVec2 r1(r0.x + contentW, r0.y + 46.0f);
                        cdl->AddRectFilled(r0, r1, ImGui::ColorConvertFloat4ToU32(pal.bgCard), 10.0f);
                        cdl->AddRect(r0, r1, ImGui::ColorConvertFloat4ToU32(pal.borderSubtle), 10.0f);
                        cdl->AddText(fHead, 14.8f, ImVec2(r0.x + 16.0f, r0.y + 14.0f),
                                     ImGui::ColorConvertFloat4ToU32(pal.textPrimary), (pname + ".cfg").c_str());

                        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
                        ImGui::SetCursorScreenPos(ImVec2(r1.x - 232.0f, r0.y + 9.0f));
                        ImGui::PushStyleColor(ImGuiCol_Button, pal.accent);
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, pal.accentSecondary);
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
                        if (ImGui::Button("Load", ImVec2(68.0f, 28.0f))) {
                            if (g_Configs.Load(pname)) {
                                g_Sound.PlayToggle(true);
                                PushToast("Profile Loaded", pname, true);
                            }
                        }
                        ImGui::PopStyleColor(3);

                        ImGui::SameLine(0.0f, 6.0f);
                        ImGui::PushStyleColor(ImGuiCol_Button, pal.bgWidget);
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, pal.bgWidgetHover);
                        ImGui::PushStyleColor(ImGuiCol_Text, pal.textPrimary);
                        if (ImGui::Button("Save", ImVec2(68.0f, 28.0f))) {
                            if (g_Configs.Save(pname)) {
                                g_Sound.PlayToggle(true);
                                PushToast("Profile Updated", pname, true);
                            }
                        }
                        ImGui::SameLine(0.0f, 6.0f);
                        if (ImGui::Button(ICON_FA_TRASH " Del", ImVec2(68.0f, 28.0f))) {
                            std::string deleted = pname;
                            if (g_Configs.Delete(deleted)) {
                                g_Sound.PlayToggle(false);
                                PushToast("Profile Deleted", deleted, false);
                                ImGui::PopStyleColor(3);
                                ImGui::PopStyleVar();
                                ImGui::PopID();
                                break;
                            }
                        }
                        ImGui::PopStyleColor(3);
                        ImGui::PopStyleVar();

                        ImGui::SetCursorPos(ImVec2(padX, rowLocal.y + 54.0f));
                        ImGui::PopID();
                    }
                    ImGui::Dummy(ImVec2(contentW, 24.0f));
                } else if (subIdx == 1) {
                    ImVec2 c0 = ImGui::GetCursorScreenPos();
                    ImVec2 c1(c0.x + contentW, c0.y + 24.0f + kRowHToggle * 4.0f);
                    cdl->AddRectFilled(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.bgCard), 11.0f);
                    cdl->AddRect(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.border), 11.0f);
                    ImGui::SetCursorPos(ImVec2(padX + 16.0f, padTop + 12.0f));
                    const float innerW = contentW - 32.0f;
                    SpringSettingToggleRow("cf_c", "Load Combat Modules",  &g_Configs.filterCombat,  innerW, pal, dt);
                    SpringSettingToggleRow("cf_v", "Load Visuals Modules", &g_Configs.filterVisuals, innerW, pal, dt);
                    SpringSettingToggleRow("cf_m", "Load Misc Modules",    &g_Configs.filterMisc,    innerW, pal, dt);
                    SpringSettingToggleRow("cf_t", "Load Theme & Audio",   &g_Configs.filterTheme,   innerW, pal, dt);
                    ImGui::SetCursorPos(ImVec2(padX, padTop + 36.0f + kRowHToggle * 4.0f));
                    ImGui::Dummy(ImVec2(contentW, 24.0f));
                } else {
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f);
                    ImGui::PushStyleColor(ImGuiCol_Button, pal.bgWidget);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, pal.bgWidgetHover);
                    ImGui::PushStyleColor(ImGuiCol_Text, pal.textPrimary);
                    if (ImGui::Button(ICON_FA_ROTATE "  Disable All Active Gameplay Modules", ImVec2(290.0f, 36.0f))) {
                        for (auto& m : g_Modules) {
                            if (m.id != "client_hud") m.enabled = false;
                        }
                        g_Sound.PlayToggle(false);
                        PushToast("Relinquish", "All gameplay modules disabled", false);
                    }
                    ImGui::SetCursorPos(ImVec2(padX, ImGui::GetCursorPosY() + 8.0f));
                    if (ImGui::Button("Reset All Module Settings To Defaults", ImVec2(290.0f, 36.0f))) {
                        InitModules();
                        g_Sound.PlayTab();
                        PushToast("Relinquish", "Restored factory defaults", true);
                    }
                    ImGui::PopStyleColor(3);
                    ImGui::SetCursorPos(ImVec2(padX, ImGui::GetCursorPosY() + 8.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.18f, 0.26f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.90f, 0.24f, 0.34f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
                    if (ImGui::Button(ICON_FA_XMARK "  Unload & Exit Relinquish", ImVec2(290.0f, 36.0f))) {
                        requestExit = true;
                    }
                    ImGui::PopStyleColor(3);
                    ImGui::PopStyleVar();
                    ImGui::Dummy(ImVec2(contentW, 24.0f));
                }
            } else if (topIdx == 4) {
                // ---------------- SETTINGS PAGE ----------------
                ImDrawList* sdl = ImGui::GetWindowDrawList();
                const float innerW = contentW - 32.0f;

                if (subIdx == 0) {
                    ImVec2 c0 = ImGui::GetCursorScreenPos();
                    ImVec2 c1(c0.x + contentW, c0.y + 66.0f);
                    sdl->AddRectFilled(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.bgCard), 11.0f);
                    sdl->AddRect(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.border), 11.0f);
                    sdl->AddText(fReg, 14.0f, ImVec2(c0.x + 16.0f, c0.y + 24.0f),
                                 ImGui::ColorConvertFloat4ToU32(pal.textSecondary), "Menu Toggle Hotkey");

                    std::string mkLbl = g_Theme.rebindMenuKey ? "Press any key..." : VkToString(g_Theme.menuToggleVk);
                    ImGui::SetCursorScreenPos(ImVec2(c1.x - 166.0f, c0.y + 16.0f));
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7.0f);
                    ImGui::PushStyleColor(ImGuiCol_Button, g_Theme.rebindMenuKey ? pal.accent : pal.bgWidget);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, pal.bgWidgetHover);
                    ImGui::PushStyleColor(ImGuiCol_Text, pal.textPrimary);
                    if (ImGui::Button(mkLbl.c_str(), ImVec2(150.0f, 34.0f))) {
                        bool nextRebind       = !g_Theme.rebindMenuKey;
                        g_Theme.rebindMenuKey = nextRebind;
                        if (nextRebind) ArmKeybindCapture(nullptr, true);
                        g_Sound.PlayTab();
                    }
                    ImGui::PopStyleColor(3);
                    ImGui::PopStyleVar();
                    ImGui::SetCursorPos(ImVec2(padX, padTop + 80.0f));
                    ImGui::Dummy(ImVec2(contentW, 24.0f));
                } else if (subIdx == 1) {
                    static const std::vector<std::string> kThemes = { "Dark Carbon", "Crisp Light", "System Sync" };
                    ImVec2 c0 = ImGui::GetCursorScreenPos();
                    float cardH = 24.0f + kRowHCombo + kRowHColor * 2.0f + 46.0f;
                    ImVec2 c1(c0.x + contentW, c0.y + cardH);
                    sdl->AddRectFilled(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.bgCard), 11.0f);
                    sdl->AddRect(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.border), 11.0f);

                    ImGui::SetCursorPos(ImVec2(padX + 16.0f, padTop + 12.0f));
                    int tm = static_cast<int>(g_Theme.themeMode);
                    if (SpringComboRow("st_theme", "Interface Theme", &tm, kThemes, innerW, pal, dt)) {
                        g_Theme.themeMode = static_cast<ThemeMode>(tm);
                    }
                    SpringColorRow("st_acc1", "Primary Accent Colour",   &g_Theme.accentPrimary, innerW, pal);
                    SpringColorRow("st_acc2", "Secondary Gradient Stop", &g_Theme.accentSecond,  innerW, pal);

                    ImGui::SetCursorPos(ImVec2(padX + 16.0f, ImGui::GetCursorPosY() + 10.0f));
                    struct SwatchPreset { const char* name; ImVec4 a; ImVec4 b; };
                    static const SwatchPreset kPresets[] = {
                        { "Relinquish Rose", ImVec4(0.95f, 0.28f, 0.40f, 1.0f), ImVec4(0.58f, 0.32f, 0.94f, 1.0f) },
                        { "CS2 Amber",       ImVec4(0.96f, 0.68f, 0.20f, 1.0f), ImVec4(0.94f, 0.36f, 0.22f, 1.0f) },
                        { "Arcane Violet",   ImVec4(0.68f, 0.44f, 0.98f, 1.0f), ImVec4(0.32f, 0.68f, 0.98f, 1.0f) },
                        { "Cyber Cyan",      ImVec4(0.22f, 0.84f, 0.96f, 1.0f), ImVec4(0.28f, 0.52f, 0.98f, 1.0f) },
                        { "Emerald Viper",   ImVec4(0.24f, 0.90f, 0.58f, 1.0f), ImVec4(0.16f, 0.68f, 0.82f, 1.0f) }
                    };
                    float btnW = std::floor((innerW - 32.0f) / 5.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7.0f);
                    for (int pi = 0; pi < 5; ++pi) {
                        if (pi > 0) ImGui::SameLine(0.0f, 8.0f);
                        ImGui::PushStyleColor(ImGuiCol_Button, kPresets[pi].a);
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kPresets[pi].b);
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
                        if (ImGui::Button(kPresets[pi].name, ImVec2(btnW, 28.0f))) {
                            g_Theme.accentPrimary = kPresets[pi].a;
                            g_Theme.accentSecond  = kPresets[pi].b;
                            g_Sound.PlayTab();
                        }
                        ImGui::PopStyleColor(3);
                    }
                    ImGui::PopStyleVar();
                    ImGui::SetCursorPos(ImVec2(padX, padTop + cardH + 12.0f));
                    ImGui::Dummy(ImVec2(contentW, 24.0f));
                } else if (subIdx == 2) {
                    static const std::vector<std::string> kAtmosOpts = { "Tactical Vignette", "CRT Scanlines", "Laser Grid", "Crimson Nebula" };
                    ImVec2 c0 = ImGui::GetCursorScreenPos();
                    float cardH = 24.0f + kRowHToggle * 2.0f + kRowHSlider * 3.0f + kRowHCombo;
                    ImVec2 c1(c0.x + contentW, c0.y + cardH);
                    sdl->AddRectFilled(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.bgCard), 11.0f);
                    sdl->AddRect(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.border), 11.0f);

                    ImGui::SetCursorPos(ImVec2(padX + 16.0f, padTop + 12.0f));
                    SpringSettingToggleRow("sh_back",  "HLSL Menu Glass Backdrop", &g_Shaders.backdropEnabled,    innerW, pal, dt);
                    SpringSliderFloatRow("sh_sheen",   "Diagonal Sheen Sweep",     &g_Shaders.sheenIntensity,     0.0f, 1.5f, "%.2f", innerW, pal, dt);
                    SpringSliderFloatRow("sh_glow",    "Dual Radial Glow",         &g_Shaders.glowIntensity,      0.0f, 1.5f, "%.2f", innerW, pal, dt);
                    SpringSettingToggleRow("sh_atmos", "Screen Atmosphere Shader", &g_Shaders.screenAtmosEnabled, innerW, pal, dt);
                    SpringComboRow("sh_mode",          "Atmosphere Preset",        &g_Shaders.screenAtmosPreset,  kAtmosOpts, innerW, pal, dt);
                    SpringSliderFloatRow("sh_alpha",   "Atmosphere Opacity",       &g_Shaders.screenAtmosAlpha,   0.05f, 0.85f, "%.2f", innerW, pal, dt);
                    ImGui::SetCursorPos(ImVec2(padX, padTop + cardH + 12.0f));
                    ImGui::Dummy(ImVec2(contentW, 24.0f));
                } else if (subIdx == 3) {
                    static const std::vector<std::string> kSpringOpts = { "Soft", "Snappy", "Bouncy", "Smooth" };
                    ImVec2 c0 = ImGui::GetCursorScreenPos();
                    float cardH = 24.0f + kRowHCombo + kRowHSlider;
                    ImVec2 c1(c0.x + contentW, c0.y + cardH);
                    sdl->AddRectFilled(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.bgCard), 11.0f);
                    sdl->AddRect(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.border), 11.0f);

                    ImGui::SetCursorPos(ImVec2(padX + 16.0f, padTop + 12.0f));
                    int sp = static_cast<int>(g_Anim.preset);
                    if (SpringComboRow("an_pre", "Spring Physics Curve", &sp, kSpringOpts, innerW, pal, dt)) {
                        g_Anim.preset = static_cast<SpringPreset>(sp);
                    }
                    SpringSliderFloatRow("an_spd", "Animation Speed Multiplier", &g_Anim.speedScale, 0.45f, 2.20f, "%.2fx", innerW, pal, dt);
                    ImGui::SetCursorPos(ImVec2(padX, padTop + cardH + 12.0f));
                    ImGui::Dummy(ImVec2(contentW, 24.0f));
                } else if (subIdx == 4) {
                    static const std::vector<std::string> kSoundOpts = { "Pop", "Click", "Bubble", "Celestial" };
                    ImVec2 c0 = ImGui::GetCursorScreenPos();
                    float cardH = 24.0f + kRowHToggle + kRowHSlider + kRowHCombo;
                    ImVec2 c1(c0.x + contentW, c0.y + cardH);
                    sdl->AddRectFilled(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.bgCard), 11.0f);
                    sdl->AddRect(c0, c1, ImGui::ColorConvertFloat4ToU32(pal.border), 11.0f);

                    ImGui::SetCursorPos(ImVec2(padX + 16.0f, padTop + 12.0f));
                    int pk = static_cast<int>(g_Sound.pack);
                    SpringSettingToggleRow("au_en", "Synthesized UI Audio", &g_Sound.enabled, innerW, pal, dt);
                    float volPct = g_Sound.volume * 100.0f;
                    if (SpringSliderFloatRow("au_vol", "Master Volume", &volPct, 0.0f, 100.0f, "%.0f%%", innerW, pal, dt)) {
                        g_Sound.volume = volPct * 0.01f;
                    }
                    if (SpringComboRow("au_pk", "Sound Pack", &pk, kSoundOpts, innerW, pal, dt)) {
                        g_Sound.pack = static_cast<SoundPack>(pk);
                        g_Sound.PlayToggle(true);
                    }
                    ImGui::SetCursorPos(ImVec2(padX, padTop + cardH + 12.0f));
                    ImGui::Dummy(ImVec2(contentW, 24.0f));
                }
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar();
    }
    ImGui::End();
    ImGui::PopStyleVar(4);
}

// ============================================================================
// 11. RELINQUISH STARTUP LOADING SPLASH VECTOR SCENE
// ============================================================================
static void RenderStartupSplashScene(ImDrawList* fgList, const Palette& pal, float sw, float sh,
                                     float progress, float splashAlpha, float timeSec) {
    ImVec2 center(sw * 0.5f, sh * 0.44f);
    float a = std::clamp(splashAlpha, 0.0f, 1.0f);

    // 1. Floating 3D Tactical ESP Wireframe Prisms in Background
    struct PrismSeed { float ox, oy, sz, spd, ph; };
    static const PrismSeed kPrisms[6] = {
        { -340.0f, -95.0f,  46.0f, 0.65f, 0.4f },
        {  355.0f, -80.0f,  52.0f, 0.55f, 1.9f },
        { -250.0f,  135.0f, 38.0f, 0.78f, 3.1f },
        {  275.0f,  125.0f, 42.0f, 0.72f, 4.4f },
        { -440.0f,  35.0f,  32.0f, 0.60f, 2.5f },
        {  445.0f,  25.0f,  34.0f, 0.68f, 5.2f }
    };
    for (const auto& pr : kPrisms) {
        float bob = std::sin(timeSec * pr.spd + pr.ph) * 12.0f;
        float rot = timeSec * 0.45f * pr.spd + pr.ph;
        ImVec2 pc(center.x + pr.ox, center.y + pr.oy + bob);
        float hs = pr.sz * 0.5f;
        float dx = std::cos(rot) * hs * 0.65f;
        float dy = std::sin(rot) * hs * 0.35f;

        ImVec2 f0(pc.x - hs, pc.y - hs * 1.35f), f1(pc.x + hs, pc.y + hs * 1.35f);
        ImVec2 b0(f0.x + dx, f0.y + dy),         b1(f1.x + dx, f1.y + dy);
        ImU32 pCol = ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, 0.24f * a));
        ImU32 sCol = ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accentSecondary.x, pal.accentSecondary.y, pal.accentSecondary.z, 0.16f * a));
        fgList->AddRect(f0, f1, pCol, 3.0f, 0, 1.2f);
        fgList->AddRect(b0, b1, sCol, 3.0f, 0, 1.0f);
        fgList->AddLine(f0, b0, sCol, 1.0f);
        fgList->AddLine(ImVec2(f1.x, f0.y), ImVec2(b1.x, b0.y), sCol, 1.0f);
        fgList->AddLine(ImVec2(f0.x, f1.y), ImVec2(b0.x, b1.y), sCol, 1.0f);
        fgList->AddLine(f1, b1, sCol, 1.0f);
    }

    // 2. Rotating Holographic Target Reticle + Twin Crossed CS2 Tactical Blades
    float hoverY = std::sin(timeSec * 2.1f) * 6.0f;
    ImVec2 heroC(center.x, center.y - 26.0f + hoverY);
    fgList->AddCircle(heroC, 92.0f, ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, 0.22f * a)), 96, 1.4f);
    fgList->AddCircle(heroC, 72.0f, ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accentSecondary.x, pal.accentSecondary.y, pal.accentSecondary.z, 0.30f * a)), 64, 1.2f);

    // Orbiting reticle ticks
    for (int k = 0; k < 4; ++k) {
        float ang = timeSec * 0.85f + k * 1.570796f;
        ImVec2 t0(heroC.x + std::cos(ang) * 84.0f, heroC.y + std::sin(ang) * 84.0f);
        ImVec2 t1(heroC.x + std::cos(ang) * 99.0f, heroC.y + std::sin(ang) * 99.0f);
        fgList->AddLine(t0, t1, ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, 0.78f * a)), 2.2f);
    }

    if (g_BrandLogo.srv) {
        float lw = 250.0f;
        float lh = lw * (static_cast<float>(g_BrandLogo.height) / std::max(1.0f, static_cast<float>(g_BrandLogo.width)));
        ImVec2 lp(heroC.x - lw * 0.5f, heroC.y - lh * 0.45f);
        fgList->AddImage(reinterpret_cast<ImTextureID>(g_BrandLogo.srv), lp, ImVec2(lp.x + lw, lp.y + lh),
                         ImVec2(0, 0), ImVec2(1, 1), IM_COL32(255, 255, 255, static_cast<int>(255 * a)));
    } else if (g_Fonts.bold28) {
        const char* title = "RELINQUISH";
        ImVec2 ts = g_Fonts.bold28->CalcTextSizeA(32.0f, FLT_MAX, 0.0f, title);
        fgList->AddText(g_Fonts.bold28, 32.0f, ImVec2(heroC.x - ts.x * 0.5f, heroC.y - 16.0f),
                        IM_COL32(250, 250, 255, static_cast<int>(255 * a)), title);
    }

    // 3. Multi-Stop Gradient Progress Bar & Stage Telemetry
    float barW = 340.0f, barH = 7.0f;
    ImVec2 bMin(center.x - barW * 0.5f, center.y + 112.0f);
    ImVec2 bMax(center.x + barW * 0.5f, center.y + 112.0f + barH);
    fgList->AddRectFilled(bMin, bMax, IM_COL32(24, 24, 32, static_cast<int>(220 * a)), 3.5f);

    float pClamped = std::clamp(progress, 0.0f, 1.0f);
    if (pClamped > 0.01f) {
        fgList->AddRectFilledMultiColor(
            bMin, ImVec2(bMin.x + barW * pClamped, bMax.y),
            ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, a)),
            ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accentSecondary.x, pal.accentSecondary.y, pal.accentSecondary.z, a)),
            ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accentSecondary.x, pal.accentSecondary.y, pal.accentSecondary.z, a)),
            ImGui::ColorConvertFloat4ToU32(ImVec4(pal.accent.x, pal.accent.y, pal.accent.z, a))
        );
    }

    const char* stage = (pClamped < 0.28f) ? "Initializing Direct3D 11 HLSL Pipeline..."
                      : (pClamped < 0.58f) ? "Binding UnityPlayer.dll Camera & Transform SIMD..."
                      : (pClamped < 0.86f) ? "Mapping Reordered IL2CPP Metadata Tables..."
                                           : "Thanks for using Relinquish";

    ImFont* fStage = (pClamped >= 0.86f && g_Fonts.bold16) ? g_Fonts.bold16 : (g_Fonts.med15 ? g_Fonts.med15 : ImGui::GetFont());
    ImVec2 stSz = fStage->CalcTextSizeA(15.0f, FLT_MAX, 0.0f, stage);
    fgList->AddText(fStage, 15.0f, ImVec2(center.x - stSz.x * 0.5f, bMax.y + 14.0f),
                    IM_COL32(236, 238, 248, static_cast<int>(240 * a)), stage);
}

// ============================================================================
// 12. DIRECT3D 11 OVERLAY WINDOW & MAIN EXECUTION LOOP
// ============================================================================
static ID3D11Device*           g_pd3dDevice           = nullptr;
static ID3D11DeviceContext*    g_pd3dDeviceContext    = nullptr;
static IDXGISwapChain*         g_pSwapChain           = nullptr;
static ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

static void CreateRenderTarget() {
    ID3D11Texture2D* pBackBuffer = nullptr;
    if (SUCCEEDED(g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer))) && pBackBuffer) {
        g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
        pBackBuffer->Release();
    }
}

static void CleanupRenderTarget() {
    if (g_mainRenderTargetView) {
        g_mainRenderTargetView->Release();
        g_mainRenderTargetView = nullptr;
    }
}

static bool CreateDeviceD3D(HWND hWnd, UINT width, UINT height) {
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount                        = 2;
    sd.BufferDesc.Width                   = width;
    sd.BufferDesc.Height                  = height;
    sd.BufferDesc.Format                  = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator   = 0;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags                              = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage                        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow                       = hWnd;
    sd.SampleDesc.Count                   = 1;
    sd.SampleDesc.Quality                 = 0;
    sd.Windowed                           = TRUE;
    sd.SwapEffect                         = DXGI_SWAP_EFFECT_DISCARD;

    D3D_FEATURE_LEVEL featureLevel{};
    const D3D_FEATURE_LEVEL levels[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                               levels, 2, D3D11_SDK_VERSION, &sd,
                                               &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (FAILED(hr)) return false;
    CreateRenderTarget();
    return true;
}

static void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain)        { g_pSwapChain->Release();        g_pSwapChain        = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice)        { g_pd3dDevice->Release();        g_pd3dDevice        = nullptr; }
}

static LRESULT WINAPI OverlayWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) return true;

    switch (msg) {
        case WM_SIZE:
            if (g_pd3dDevice != nullptr && wParam != SIZE_MINIMIZED) {
                CleanupRenderTarget();
                g_pSwapChain->ResizeBuffers(0, static_cast<UINT>(LOWORD(lParam)), static_cast<UINT>(HIWORD(lParam)), DXGI_FORMAT_UNKNOWN, 0);
                CreateRenderTarget();
            }
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU) return 0;
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

static fs::path GetExecutableDir() {
    wchar_t buf[MAX_PATH]{};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(buf).parent_path();
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR, int) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    HRESULT hrCom = CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    fs::path baseDir  = GetExecutableDir();
    fs::path assetDir = baseDir / "assets";
    if (!fs::exists(assetDir) && fs::exists(fs::current_path() / "assets")) {
        baseDir  = fs::current_path();
        assetDir = baseDir / "assets";
    }

    int initW = GetSystemMetrics(SM_CXSCREEN);
    int initH = GetSystemMetrics(SM_CYSCREEN);

    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(WNDCLASSEXW);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = OverlayWndProc;
    wc.hInstance     = hInstance;
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"RelinquishOverlayWnd";
    RegisterClassExW(&wc);

    DWORD exStyleInteractive = WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TOOLWINDOW;
    DWORD exStylePassthrough = WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE;

    HWND overlayHwnd = CreateWindowExW(exStyleInteractive, wc.lpszClassName, L"Relinquish",
                                       WS_POPUP, 0, 0, initW, initH, nullptr, nullptr, hInstance, nullptr);
    if (!overlayHwnd) {
        if (SUCCEEDED(hrCom)) CoUninitialize();
        return 1;
    }

    SetLayeredWindowAttributes(overlayHwnd, 0, 255, LWA_ALPHA);
    MARGINS margins = { -1, -1, -1, -1 };
    DwmExtendFrameIntoClientArea(overlayHwnd, &margins);

    if (!CreateDeviceD3D(overlayHwnd, static_cast<UINT>(initW), static_cast<UINT>(initH))) {
        CleanupDeviceD3D();
        DestroyWindow(overlayHwnd);
        UnregisterClassW(wc.lpszClassName, hInstance);
        if (SUCCEEDED(hrCom)) CoUninitialize();
        return 1;
    }

    ShowWindow(overlayHwnd, SW_SHOWDEFAULT);
    UpdateWindow(overlayHwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigErrorRecoveryEnableAssert  = false;
    io.ConfigErrorRecoveryEnableTooltip = false;

    g_Fonts.Load(io, assetDir);
    g_BrandLogo.LoadFromPNG(g_pd3dDevice, g_pd3dDeviceContext, assetDir / "logo.png");
    g_Shaders.Init(g_pd3dDevice);
    g_Sound.Init();
    InitModules();
    g_Configs.Init(baseDir);

    ImGui_ImplWin32_Init(overlayHwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    g_Scanner.Start();
    g_Sound.PlayStartupChime();

    bool menuOpen        = true;
    bool menuToggleHeld  = false;
    bool currentlyClickThrough = false;
    bool requestExit     = false;
    int  curOverlayX     = 0, curOverlayY = 0, curOverlayW = initW, curOverlayH = initH;

    auto startClock = std::chrono::steady_clock::now();
    auto prevClock  = startClock;
    constexpr float kSplashDuration = 2.95f;

    MSG msg{};
    while (!requestExit) {
        while (PeekMessageW(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            if (msg.message == WM_QUIT) requestExit = true;
        }
        if (requestExit) break;

        auto nowClock = std::chrono::steady_clock::now();
        float dt      = std::clamp(std::chrono::duration<float>(nowClock - prevClock).count(), 0.0005f, 0.05f);
        double nowSec = std::chrono::duration<double>(nowClock - startClock).count();
        prevClock     = nowClock;

        // Menu toggle hotkey (Insert by default, configurable in Settings -> Menu)
        bool wasRebindingMenu = g_Theme.rebindMenuKey;
        PollModuleKeybinds(menuOpen);
        if (!wasRebindingMenu && !g_Theme.rebindMenuKey) {
            bool toggleDown = (GetAsyncKeyState(g_Theme.menuToggleVk) & 0x8000) != 0;
            if (toggleDown && !menuToggleHeld) {
                menuOpen = !menuOpen;
                g_Sound.PlayToggle(menuOpen);
                if (menuOpen) {
                    SetForegroundWindow(overlayHwnd);
                }
            }
            menuToggleHeld = toggleDown;
        } else {
            menuToggleHeld = true;
        }

        Module* mFreeCur = FindModule("free_cursor");
        bool freeCursorActive = (mFreeCur && mFreeCur->enabled);
        bool splashActive     = (nowSec < kSplashDuration + 0.45);

        WorldSnapshot world = g_Scanner.GetSnapshot();

        // Sync overlay rectangle to CombatMaster.exe client viewport when attached, else primary screen
        int targetX = 0, targetY = 0, targetW = GetSystemMetrics(SM_CXSCREEN), targetH = GetSystemMetrics(SM_CYSCREEN);
        bool gameForeground = true;
        if (world.attached && world.gameHwnd && IsWindow(world.gameHwnd) && !IsIconic(world.gameHwnd)) {
            RECT cr{};
            POINT tl{ 0, 0 };
            if (GetClientRect(world.gameHwnd, &cr) && ClientToScreen(world.gameHwnd, &tl)) {
                int cw = cr.right - cr.left;
                int ch = cr.bottom - cr.top;
                if (cw >= 320 && ch >= 240) {
                    targetX = tl.x;
                    targetY = tl.y;
                    targetW = cw;
                    targetH = ch;
                }
            }
            HWND fg = GetForegroundWindow();
            DWORD fgPid = 0;
            if (fg) GetWindowThreadProcessId(fg, &fgPid);
            gameForeground = (fg == nullptr || fg == world.gameHwnd || fg == overlayHwnd || fgPid == world.pid);
        }

        if (targetX != curOverlayX || targetY != curOverlayY || targetW != curOverlayW || targetH != curOverlayH) {
            curOverlayX = targetX; curOverlayY = targetY; curOverlayW = targetW; curOverlayH = targetH;
            SetWindowPos(overlayHwnd, HWND_TOPMOST, curOverlayX, curOverlayY, curOverlayW, curOverlayH, SWP_NOACTIVATE);
        }

        // Dynamic WS_EX_TRANSPARENT passthrough toggle so mouse/keyboard hit the game seamlessly when menu is closed
        bool wantInteractive = (menuOpen || freeCursorActive) && !splashActive;
        if (wantInteractive == currentlyClickThrough) {
            LONG_PTR curEx = GetWindowLongPtrW(overlayHwnd, GWL_EXSTYLE);
            if (wantInteractive) {
                SetWindowLongPtrW(overlayHwnd, GWL_EXSTYLE, (curEx & ~(WS_EX_TRANSPARENT | WS_EX_NOACTIVATE)) | exStyleInteractive);
            } else {
                SetWindowLongPtrW(overlayHwnd, GWL_EXSTYLE, curEx | exStylePassthrough);
            }
            SetWindowPos(overlayHwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED);
            currentlyClickThrough = !wantInteractive;
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        Palette pal = g_Theme.BuildPalette();
        float sw = static_cast<float>(curOverlayW);
        float sh = static_cast<float>(curOverlayH);

        float splashAlpha = 0.0f;
        if (splashActive) {
            float rawFade = (nowSec < kSplashDuration) ? 1.0f : std::max(0.0f, 1.0f - static_cast<float>((nowSec - kSplashDuration) / 0.45));
            splashAlpha = rawFade * rawFade * (3.0f - 2.0f * rawFade);
        }

        float menuAlpha = g_Anim.StepKey("ui.menu_alpha", (!splashActive && menuOpen) ? 1.0f : 0.0f, dt);
        ImVec2 menuRectMin(0, 0), menuRectMax(0, 0);

        if (splashAlpha > 0.01f) {
            float prog = std::clamp(static_cast<float>(nowSec) / (kSplashDuration * 0.88f), 0.0f, 1.0f);
            RenderStartupSplashScene(ImGui::GetForegroundDrawList(), pal, sw, sh, prog, splashAlpha, static_cast<float>(nowSec));
        } else {
            if (freeCursorActive && mFreeCur->GetBool("dim_bg", false)) {
                ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), ImVec2(sw, sh), IM_COL32(0, 0, 0, 85));
            }
            if (gameForeground || menuOpen || freeCursorActive) {
                RunCombatAndVisuals(ImGui::GetBackgroundDrawList(), world, pal, sw, sh, dt, nowSec, menuOpen, freeCursorActive);
                RenderClientHudAndToasts(ImGui::GetForegroundDrawList(), world, pal, sw, sh, dt);
            }
            if (menuAlpha > 0.01f) {
                RenderRelinquishMenu(world, pal, sw, sh, dt, menuAlpha, requestExit, menuRectMin, menuRectMax);
            }
        }

        ImGui::Render();

        const float clearCol[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clearCol);

        // Hardware D3D11 HLSL pre-ImGui passes (Loading Sky & Screen Atmosphere; Menu Backdrop runs inside WindowDrawList)
        g_Shaders.cachedCtx = g_pd3dDeviceContext;
        g_Shaders.cachedRtv = g_mainRenderTargetView;
        if (g_Shaders.ready) {
            ShaderParamsCB cb{};
            cb.screenSize[0] = sw;
            cb.screenSize[1] = sh;
            cb.time          = static_cast<float>(nowSec);
            std::memcpy(cb.accentA, &pal.accent,          sizeof(float) * 4);
            std::memcpy(cb.accentB, &pal.accentSecondary, sizeof(float) * 4);
            std::memcpy(cb.bgBase,  &pal.bgWindow,        sizeof(float) * 4);

            if (splashAlpha > 0.01f) {
                cb.alpha = splashAlpha;
                g_Shaders.DrawFullscreenPass(g_pd3dDeviceContext, g_mainRenderTargetView, g_Shaders.psLoading, cb);
            } else if (g_Shaders.screenAtmosEnabled && (gameForeground || menuOpen)) {
                cb.alpha = g_Shaders.screenAtmosAlpha;
                cb.mode  = g_Shaders.screenAtmosPreset;
                g_Shaders.DrawFullscreenPass(g_pd3dDeviceContext, g_mainRenderTargetView, g_Shaders.psAtmos, cb);
            }
        }

        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_pSwapChain->Present(1, 0);
    }

    g_Scanner.Stop();
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    g_Sound.Shutdown();
    g_Shaders.Shutdown();
    g_BrandLogo.Shutdown();
    CleanupDeviceD3D();

    DestroyWindow(overlayHwnd);
    UnregisterClassW(wc.lpszClassName, hInstance);
    if (SUCCEEDED(hrCom)) CoUninitialize();
    return 0;
}
