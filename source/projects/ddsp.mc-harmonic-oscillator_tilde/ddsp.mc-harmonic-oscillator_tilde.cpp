/// @file
/// @ingroup     minexamples
/// @copyright    Copyright 2018 The Min-DevKit Authors. All rights reserved.
/// @license    Use of this source code is governed by the MIT License found in the License.md file.

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
namespace max = c74::max;
using namespace torch::indexing;

class ddsp_mc_harmonic_oscillator_tilde : public object<ddsp_mc_harmonic_oscillator_tilde>, public mc_operator<> {
public:
    MIN_DESCRIPTION     { "DDSP MC harmonic oscillator~." };
    MIN_TAGS            { "DDSP" };
    MIN_AUTHOR          { "Cycling '74" };
    MIN_RELATED         { "" };
    
    inlet<> in_pitch                { this, "(signal) fundamental frequency" };
    inlet<> in_harmonic_amplitudes  { this, "(multichannelsignal) harmonic amplitudes" };
    outlet<> out_signal             { this, "(multichannelsignal) harmonic signals", "multichannelsignal" };
    
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
        
        // remove harmonics above nyquist
        auto fundamental_frequencies = torch::mul(fundamental_frequency_tensor, torch::reshape(torch::arange(1, m_num_harmonics + 1), {1, m_num_harmonics}));
        harmonic_amplitudes_tensor = torch::where(fundamental_frequencies < m_samplerate/2.0, harmonic_amplitudes_tensor, 0.0);
        // harmonic_amplitudes_tensor = harmonic_amplitudes_tensor + 0.0001;
        
        // compute instantaneous phase and save initial phases
        auto omega = torch::cumsum(2 * M_PI * m_one_over_samplerate * fundamental_frequency_tensor, 0);
        omega = torch::add(omega, m_phase);
        m_phase = omega[signal_vector_size-1] % (2 * M_PI);
        auto omega_harmonics = torch::mul(omega, torch::arange(1, m_num_harmonics + 1).to(omega));
        
        // generate harmonic sinusoids
        auto harmonic_signals = torch::mul(torch::sin(omega_harmonics), harmonic_amplitudes_tensor);
        harmonic_signals = harmonic_signals.permute({1, 0});
        harmonic_signals = torch::flatten(harmonic_signals, 1);
        
        // copy into harmonic signals into according channels
        auto harmonic_signal_ptr = harmonic_signals.contiguous().data_ptr<double>();
        for (int harmonic = 0; harmonic < MAX_NUM_HARMONICS; harmonic++) {
            int offset = harmonic * signal_vector_size;
            memcpy(output.samples(harmonic), harmonic_signal_ptr + offset, signal_vector_size * sizeof(double));
        }
    }
        
    // when object is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "DDSP MC harmonic oscillator~ loaded" << endl;
            auto class_ptr = args[0];
            max::class_addmethod(class_ptr, (max::method)set_out_channels, "multichanneloutputs", max::A_CANT, 0);
            return {};
        }
    };
    
    static long set_out_channels(void* obj, long outletindex) {
        return MAX_NUM_HARMONICS;
    }

private:
    double m_samplerate;
    double m_one_over_samplerate;
    
    double* harmonic_amplitudes = NULL;
    torch::Tensor m_phase = torch::zeros({1});
    int m_num_harmonics;
    
};


MIN_EXTERNAL(ddsp_mc_harmonic_oscillator_tilde);

