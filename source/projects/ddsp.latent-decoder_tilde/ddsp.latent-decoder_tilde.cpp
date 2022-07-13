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

#define B_SIZE 1024
#define N_HARMONICS 100
#define N_MAGNITUDES 65
#define N_MFCCS 3

using namespace c74::min;
namespace max = c74::max;
//using namespace std::chrono;

class DDSPModel
{
public:
    DDSPModel();
    int load(std::string path);
    void perform(double *pitch, double *loudness, double *mfccs, double *f0, double *harmonic_amplitudes, double *filter_magnitudes, int buffer_size);

private:
    torch::jit::script::Module m_scripted_model;
    int m_model_is_loaded;
};

DDSPModel::DDSPModel() : m_model_is_loaded(0)
{
    at::init_num_threads();
}

int DDSPModel::load(std::string path)
{
    try
    {
        m_scripted_model = torch::jit::load(path);
        m_scripted_model.eval();
        m_model_is_loaded = 1;
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        m_model_is_loaded = 0;
        return 0;
    }
}

void DDSPModel::perform(double *pitch, double *loudness, double *mfccs, double *f0, double *harmonic_amplitudes, double *filter_magnitudes, int buffer_size)
{
    torch::NoGradGuard no_grad;
    if (m_model_is_loaded)
    {
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        auto pitch_tensor = torch::from_blob(pitch, {1, buffer_size, 1}, options);
        auto loudness_tensor = torch::from_blob(loudness, {1, buffer_size, 1}, options);
        auto mfcc_tensor = torch::from_blob(mfccs, {1, N_MFCCS, buffer_size}, options);
        mfcc_tensor = mfcc_tensor.permute({0, 2, 1});

        std::vector<torch::jit::IValue> inputs = {pitch_tensor.to(torch::kFloat32), loudness_tensor.to(torch::kFloat32), mfcc_tensor.to(torch::kFloat32)};
        
//        auto start_time = high_resolution_clock::now();
        auto out_tensor = m_scripted_model.forward(inputs);
//        auto stop_time = high_resolution_clock::now();
//        auto duration = duration_cast<microseconds>(stop_time - start_time);
//        std::cout << duration.count() << std::endl;
        auto harmonic_amplitudes_tensor = out_tensor.toTuple()->elements()[0].toTensor().to(torch::kFloat64);
        auto filter_magnitudes_tensor = out_tensor.toTuple()->elements()[1].toTensor().to(torch::kFloat64);

        // upsampling / interpolation
        namespace F = torch::nn::functional;
        pitch_tensor = pitch_tensor.contiguous();
        pitch_tensor = pitch_tensor.permute({0, 2, 1});
        pitch_tensor = F::interpolate(pitch_tensor, F::InterpolateFuncOptions()
                                                    .size(std::vector<int64_t>{buffer_size})
                                                    .mode(torch::kNearest)); // originally DDSP uses linear method
        pitch_tensor = pitch_tensor.permute({0, 2, 1});
        pitch_tensor = torch::flatten(pitch_tensor);
        
        harmonic_amplitudes_tensor = harmonic_amplitudes_tensor.contiguous();
        harmonic_amplitudes_tensor = harmonic_amplitudes_tensor.permute({0, 2, 1}); // permute: switch rows and columns, rows = harmonics, columns = samples
        harmonic_amplitudes_tensor = F::interpolate(harmonic_amplitudes_tensor, F::InterpolateFuncOptions()
                                                    .size(std::vector<int64_t>{buffer_size})
                                                    .mode(torch::kNearest)); // originally DDSP uses window method
        harmonic_amplitudes_tensor = torch::flatten(harmonic_amplitudes_tensor, 1);
        
        filter_magnitudes_tensor = filter_magnitudes_tensor.contiguous();
        filter_magnitudes_tensor = filter_magnitudes_tensor.permute({0, 2, 1});
        filter_magnitudes_tensor = F::interpolate(filter_magnitudes_tensor, F::InterpolateFuncOptions()
                                                    .size(std::vector<int64_t>{buffer_size})
                                                    .mode(torch::kNearest));
        filter_magnitudes_tensor = torch::flatten(filter_magnitudes_tensor, 1);
        
        // Copy to output buffers
        auto pitch_ptr = pitch_tensor.contiguous().data_ptr<double>();
        memcpy(f0, pitch_ptr, buffer_size * sizeof(double));
        
        auto harmonic_amplitudes_ptr = harmonic_amplitudes_tensor.contiguous().data_ptr<double>();
        memcpy(harmonic_amplitudes, harmonic_amplitudes_ptr, buffer_size * N_HARMONICS * sizeof(double));
        
        auto filter_magnitudes_ptr = filter_magnitudes_tensor.contiguous().data_ptr<double>();
        memcpy(filter_magnitudes, filter_magnitudes_ptr, buffer_size * N_MAGNITUDES * sizeof(double));
    }
}

class ddsp_latent_decoder_tilde : public object<ddsp_latent_decoder_tilde>, public mc_operator<> {
public:
    MIN_DESCRIPTION	{ "DDSP latent decoder~." };
    MIN_TAGS		{ "DDSP" };
    MIN_AUTHOR		{ "Cycling '74" };
    MIN_RELATED		{ "" };

    // execute the ddsp computation in a separate thread
    void thread_perform(double *pitch, double *loudness, double *mfccs, double *f0, double *harmonic_amplitudes, double *filter_magnitudes, int buffer_size)
    {
        m_model->perform(pitch, loudness, mfccs, f0, harmonic_amplitudes, filter_magnitudes, buffer_size);
    }

    // constructor
    ddsp_latent_decoder_tilde(const atoms& args = {}) {
        // to ensure safety in possible attribute settings
        m_model = new DDSPModel;
            
        // configure inlets and outlets
        auto in_pitch_frequency = std::make_unique<inlet<>>(this, "(signal) pitch frequency");
        auto in_loudness = std::make_unique<inlet<>>(this, "(signal) loudness");
        auto in_mfccs = std::make_unique<inlet<>>(this, "(multichannelsignal) latent mfccs");
        auto out_fundamental_frequency = std::make_unique<outlet<>>(this, "(multichannelsignal) fundamental frequency", "multichannelsignal");
        auto out_harmonic_amplitudes = std::make_unique<outlet<>>(this, "(multichannelsignal) harmonic amplitudes", "multichannelsignal");
        auto out_filter_magnitudes = std::make_unique<outlet<>>(this, "(multichannelsignal) filter magnitudes", "multichannelsignal");
        
        m_inlets.push_back( std::move(in_pitch_frequency) );
        m_inlets.push_back( std::move(in_loudness) );
        m_inlets.push_back( std::move(in_mfccs) );
        m_outlets.push_back( std::move(out_fundamental_frequency) );
        m_outlets.push_back( std::move(out_harmonic_amplitudes) );
        m_outlets.push_back( std::move(out_filter_magnitudes) );
    
        cout << "Model loaded successfully" << endl;
    }
    
    void operator()(audio_bundle input, audio_bundle output) {
        
        int n = input.frame_count();
        
        // Copy pitch and loudness into input buffers
        memcpy(m_pitch_buffer + m_head, input.samples(0), n * sizeof(double));
        memcpy(m_loudness_buffer + m_head, input.samples(1), n * sizeof(double));
        
        // Copy f0 into output
        memcpy(output.samples(0), m_f0_buffer + m_head, output.frame_count() * sizeof(double));
        
        // Calculate offset for mc buffers
        int offset_mfccs = m_head % B_SIZE;
        int offset_harmonics = m_head % B_SIZE;
        int offset_magnitudes = m_head % B_SIZE;
        if (m_head >= B_SIZE) { // point to next part of the buffer
            offset_mfccs += N_MFCCS * B_SIZE;
            offset_harmonics += N_HARMONICS * B_SIZE;
            offset_magnitudes += N_MAGNITUDES * B_SIZE;
        }
        
        // Copy mfccs into input buffers
        m_num_mfccs = input.channel_count() - 2;
        for (int mfcc = 0; mfcc < m_num_mfccs; ++mfcc) {
            int ch = mfcc + 2;
            memcpy(m_mfcc_buffer + offset_mfccs, input.samples(ch), n * sizeof(double));
            offset_mfccs += B_SIZE;
        }
        
        // Copy harmonics into output channels
        for (int harmonic = 0; harmonic < N_HARMONICS; harmonic++) {
            int ch = harmonic + 1;
            memcpy(output.samples(ch), m_harmonic_amplitudes_buffer + offset_harmonics, output.frame_count() * sizeof(double));
            offset_harmonics += B_SIZE;
        }
        
        // Copy filter magnitudes into output channels
        for (int magnitude = 0; magnitude < N_MAGNITUDES; magnitude++) {
            int ch = magnitude + N_HARMONICS + 1;
            memcpy(output.samples(ch), m_filter_magnitudes_buffer + offset_magnitudes, output.frame_count() * sizeof(double));
            offset_magnitudes += B_SIZE;
        }
        
        m_head += n; // progress with the m_head
        
        if (!(m_head % B_SIZE)) { // if it is B_SIZE or B_SIZE * 2
            if (m_compute_thread) {
                m_compute_thread->join();
            }
            m_model_head = ((m_head + B_SIZE) % (2 * B_SIZE)); // points to the next / previous B_SIZE spaces available
            m_compute_thread = new std::thread(&ddsp_latent_decoder_tilde::thread_perform, this,
                                            m_pitch_buffer + m_model_head,
                                            m_loudness_buffer + m_model_head,
                                            m_mfcc_buffer + N_MFCCS * m_model_head,
                                            m_f0_buffer + m_model_head,
                                            m_harmonic_amplitudes_buffer + N_HARMONICS * m_model_head,
                                            m_filter_magnitudes_buffer + N_MAGNITUDES * m_model_head,
                                            B_SIZE); // compute the buffers in a separate thread

            m_head = m_head % (2 * B_SIZE); // set the m_head to the next available value
        }
    }
        
    // when object is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "DDSP latent decoder~ loaded" << endl;
            auto class_ptr = args[0];
//            auto c = max::class_findbyname(max::gensym("box"), max::gensym("ddsp.decoder_controls~"));
            max::class_addmethod(class_ptr, (max::method)set_out_channels, "multichanneloutputs", max::A_CANT, 0);
            return {};
        }
    };
    
    message<> load { this, "load", "Load control model (*.ts).",
        MIN_FUNCTION {
            if(max::open_dialog(m_filename, &m_path, &m_type, &m_types[0], static_cast<short>(m_types.size())) == 0)
            {
                cout << "Loading model..." << endl;
                max::path_toabsolutesystempath(m_path, m_filename, m_model_path);
                int model_is_loaded = m_model->load(m_model_path); // try to load the model
                if (model_is_loaded) { // if loaded correctly
                    cout << "Model loaded successfully" << endl;
                }
                else {
                    cerr << "Error loading model" << endl;
                }
            }
            return {};
        }
    };
        
    static long set_out_channels(void* obj, long outletindex) {
        int num_channels = 1; // f0
        if (outletindex == 1)
            num_channels = 100; // harmonic amplitudes
        if (outletindex == 2)
            num_channels = 65; // filter mangitudes
        return num_channels;
    }


private:
    // inlets and outlets that will be defined at runtime
    std::vector< std::unique_ptr<inlet<>> > m_inlets;
    std::vector< std::unique_ptr<outlet<>> > m_outlets;

    // controller variables for the model
    DDSPModel *m_model { nullptr };
    
    const static int m_num_harmonics = 100;

    // buffers
    double m_pitch_buffer[2 * B_SIZE];
    double m_loudness_buffer[2 * B_SIZE];
    double m_mfcc_buffer[2 * B_SIZE * N_MFCCS] = {};
    double m_f0_buffer[2 * B_SIZE];
    double m_harmonic_amplitudes_buffer[2 * B_SIZE * N_HARMONICS];
    double m_filter_magnitudes_buffer[2 * B_SIZE * N_MAGNITUDES];
    
    int m_num_mfccs;

    // variable to store the position in the buffer
    int m_head {0};
    int m_model_head {0};

    // pointer to run a different thread for the ddsp computation
    std::thread *m_compute_thread;
    
    // model path
    short m_path {};
    char m_filename[MAX_FILENAME_CHARS] {};
    char m_model_path[MAX_PATH_CHARS] {};
    max::t_fourcc m_type {};
    std::vector<max::t_fourcc> m_types {};
};


MIN_EXTERNAL(ddsp_latent_decoder_tilde);
