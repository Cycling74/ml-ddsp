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

class ddsp_harmonic_oscillator_tilde : public object<ddsp_harmonic_oscillator_tilde>, public mc_operator<> {
public:
    MIN_DESCRIPTION	{ "DDSP harmonic oscillator~." };
    MIN_TAGS		{ "DDSP" };
    MIN_AUTHOR		{ "Cycling '74" };
    MIN_RELATED		{ "" };
    
    inlet<> in_pitch { this, "(signal) fundamental frequency" };
    inlet<> in_harmonic_amplitudes { this, "(multichannelsignal) harmonic amplitudes" };
    outlet<> out_signal        { this, "(signal) audio output", "signal", };
    
    message<> dspsetup { this, "dspsetup",
        MIN_FUNCTION {
            m_samplerate = (float)args[0];
            m_one_over_samplerate = 1.0 / m_samplerate;
            return {};
        }
    };
    
    void operator()(audio_bundle input, audio_bundle output) {
        m_num_harmonics = input.channel_count() - 1;
        int signal_vector_size = input.frame_count();
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        auto zero_tensor = torch::zeros({signal_vector_size, m_num_harmonics});
        
        // read input channels and convert to tensors
        auto inputs = torch::from_blob(input.samples(0), {signal_vector_size, 1}, options); // ch 0: f0, ch 1-101: harmonics
        for (int i = 0; i < m_num_harmonics; ++i) {
            auto harmonic_amplitudes = torch::from_blob(input.samples(i+1), {signal_vector_size, 1}, options);
            inputs = torch::cat({inputs, harmonic_amplitudes}, 1);
        }
        
        auto fundamental_frequency_tensor = inputs.index({"...", Slice(None, 1)});
        auto harmonic_amplitudes_tensor = inputs.index({"...", Slice(1, None)});
        
        // compute instantaneous phase and save initial phases
        auto omega = torch::cumsum(2 * M_PI * m_one_over_samplerate * fundamental_frequency_tensor, 1);
        omega = torch::add(omega, m_phase);
        m_phase = omega[signal_vector_size-1] % (2 * M_PI);
        auto omega_harmonics = torch::mul(omega, torch::arange(m_num_harmonics).to(omega));
        
        // generate harmonic sinusoids and add
        auto harmonic_signal = torch::sum(1 * torch::sin(omega_harmonics), 1); // TODO: incorporate total amplitude (1)
        
        // copy to outlet
        auto harmonic_signal_ptr = harmonic_signal.contiguous().data_ptr<double>();
        memcpy(output.samples(0), harmonic_signal_ptr, signal_vector_size * sizeof(double));
    }
        
    // when object is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "DDSP harmonic oscillator~ loaded" << endl;
            return {};
        }
    };

private:
    double m_samplerate;
    double m_one_over_samplerate;
    
    torch::Tensor m_phase = torch::zeros({1});
    int m_num_harmonics;
    
};


MIN_EXTERNAL(ddsp_harmonic_oscillator_tilde);
