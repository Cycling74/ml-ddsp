/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "dlfcn.h"
#include <string>
#include <thread>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <iostream>
#include <vector>
#include <stdlib.h>

using namespace c74::min;
using namespace torch::indexing;

class ddsp_filtered_noise_tilde : public object<ddsp_filtered_noise_tilde>, public mc_operator<> {
public:
    MIN_DESCRIPTION	{ "DDSP harmonic oscillator~." };
    MIN_TAGS		{ "DDSP" };
    MIN_AUTHOR		{ "Cycling '74" };
    MIN_RELATED		{ "" };
    
    inlet<> in_filter_magnitudes    { this, "(multichannelsignal) filter magnitudes" };
    outlet<> out_signal             { this, "(signal) audio output", "signal", };
    
    message<> dspsetup { this, "dspsetup",
        MIN_FUNCTION {
            m_samplerate = (float)args[0];
            m_one_over_samplerate = 1.0 / m_samplerate;
            return {};
        }
    };
    
    void operator()(audio_bundle input, audio_bundle output) {
        m_num_magnitudes = input.channel_count();
        int signal_vector_size = input.frame_count();
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        auto zero_tensor = torch::zeros({signal_vector_size, 1});
        
        // read input channels and convert to tensors // TODO: initialize a 2D zero tensor and assign columns via torch::indexing
        auto inputs = zero_tensor;
        for (int i = 0; i < m_num_magnitudes; ++i) {
            auto filter_magnitudes = torch::from_blob(input.samples(i), {signal_vector_size, 1}, options);
            inputs = torch::cat({inputs, filter_magnitudes}, -1);
        }

        auto filter_magnitudes_tensor = inputs.index({"...", Slice(1, None)});
        
        // process one frequency response per signal vector
        filter_magnitudes_tensor = filter_magnitudes_tensor.index({Slice(None, 1), "..."});
        
        // frequency sampling method
        filter_magnitudes_tensor = torch::stack({filter_magnitudes_tensor, torch::zeros_like(filter_magnitudes_tensor)}, -1);
        filter_magnitudes_tensor = torch::view_as_complex(filter_magnitudes_tensor);
        
        // retrieve time-domain IR from filter magnitudes
        auto impulse_response = torch::fft::irfft(filter_magnitudes_tensor);
        int impulse_response_size = torch::size(impulse_response, -1);
        
        // shift zero-frequency component to center of the spectrum to retrieve an even IR
        // even IR is necessary to design a zero-phase filter that has the advantage of no phase distortion
        // pseudo real-time because zero-phase filters are non-causal
        impulse_response = torch::roll(impulse_response, (int)floor(impulse_response_size * 0.5), -1);
        
        // use hann window to time-limit desired impulse response
        auto window = torch::hann_window(impulse_response_size, options);
        impulse_response = torch::mul(impulse_response, window);
//        namespace F = torch::nn::functional;
//        int target_size = 256; // TODO: change to signal vector size
//        auto pad_options = F::PadFuncOptions({0, (target_size - impulse_response_size)}).mode(torch::kConstant);
//        impulse_response = torch::nn::functional::pad(impulse_response, pad_options);
        
        // shift back IR to causal form
        impulse_response = torch::roll(impulse_response, -(int)floor(impulse_response_size * 0.5), -1);
        
        // generate noise signal
        auto noise = torch::rand({1, impulse_response_size}).to(impulse_response) * 2 - 1;
        
        // apply non-windowed non-overlapping STFT/ISTFT to efficiently compute convolution for large impulse response sizes
        namespace F = torch::nn::functional;
        auto pad_options = F::PadFuncOptions({impulse_response_size, 0}).mode(torch::kConstant);
        impulse_response = torch::nn::functional::pad(impulse_response, pad_options);
        pad_options = F::PadFuncOptions({0, impulse_response_size}).mode(torch::kConstant);
        noise = torch::nn::functional::pad(noise, pad_options);
        
        auto filtered_noise = torch::fft::irfft(torch::fft::rfft(noise) * torch::fft::rfft(impulse_response));
        
        // compensate for the group delay of the filter by trimming the front
        // the group delay is constant because the filter is linear phase
        filtered_noise = filtered_noise.index({"...", Slice((int)floor(filtered_noise.size(-1) * 0.5), None)});
        
        // copy to outlet
        auto filtered_noise_ptr = filtered_noise.contiguous().data_ptr<double>();
        memcpy(output.samples(0), filtered_noise_ptr, signal_vector_size * sizeof(double)); // output.frame_count() ?
    }
        
    // when object is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "DDSP filtered noise~ loaded" << endl;
            return {};
        }
    };

private:
    double m_samplerate;
    double m_one_over_samplerate;
    
    torch::Tensor m_phase = torch::zeros({1});
    int m_num_magnitudes;
    
};


MIN_EXTERNAL(ddsp_filtered_noise_tilde);
