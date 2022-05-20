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

#define MAX_NUM_HARMONICS 100

using namespace c74::min;
using namespace torch::indexing;

class ddsp_harmonic_oscillator_tilde : public object<ddsp_harmonic_oscillator_tilde>, public mc_operator<> {
public:
    MIN_DESCRIPTION	{ "DDSP harmonic oscillator~." };
    MIN_TAGS		{ "DDSP" };
    MIN_AUTHOR		{ "Cycling '74" };
    MIN_RELATED		{ "" };
    
    inlet<> in_pitch                { this, "(signal) fundamental frequency" };
    inlet<> in_harmonic_amplitudes  { this, "(multichannelsignal) harmonic amplitudes" };
    outlet<> out_signal             { this, "(signal) audio output", "signal", };
    
    message<> dspsetup { this, "dspsetup",
        MIN_FUNCTION {
            m_samplerate = (float)args[0];
            m_one_over_samplerate = 1.0 / m_samplerate;
            int vector_size = args[1];
            delete [] harmonic_amplitudes;
            harmonic_amplitudes = new double[MAX_NUM_HARMONICS * vector_size];
            memset(harmonic_amplitudes, 0, MAX_NUM_HARMONICS * vector_size * sizeof(*harmonic_amplitudes));
            return {};
        }
    };
    
    void operator()(audio_bundle input, audio_bundle output) {
        m_num_harmonics = input.channel_count() - 1;
        int signal_vector_size = input.frame_count();
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        
        // read input channels and convert to tensors (ch 0: f0, ch 1-101: harmonics)
        auto fundamental_frequency_tensor = torch::from_blob(input.samples(0), {signal_vector_size, 1}, options);
        
        for (int i = 0; i < m_num_harmonics; ++i) {
            memcpy(harmonic_amplitudes + i*signal_vector_size, input.samples(i+1), signal_vector_size * sizeof(double));
        }

        auto harmonic_amplitudes_tensor = torch::from_blob(harmonic_amplitudes, {MAX_NUM_HARMONICS, signal_vector_size}, options);
        harmonic_amplitudes_tensor = harmonic_amplitudes_tensor.permute({1, 0});
        
        // compute instantaneous phase and save initial phases
        auto omega = torch::cumsum(2 * M_PI * m_one_over_samplerate * fundamental_frequency_tensor, 0);
        omega = torch::add(omega, m_phase);
        m_phase = omega[signal_vector_size-1] % (2 * M_PI);
        auto omega_harmonics = torch::mul(omega, torch::arange(1, m_num_harmonics + 1).to(omega));
        
        // generate harmonic sinusoids and add
        auto harmonic_signal = torch::sum(torch::mul(torch::sin(omega_harmonics), harmonic_amplitudes_tensor), -1);
        
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
    
    double* harmonic_amplitudes = NULL;
    torch::Tensor m_phase = torch::zeros({1});
    int m_num_harmonics;
    
};


MIN_EXTERNAL(ddsp_harmonic_oscillator_tilde);
