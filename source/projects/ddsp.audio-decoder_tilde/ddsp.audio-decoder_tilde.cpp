/// @file
/// @ingroup    minexamples
/// @copyright  Copyright 2018 The Min-DevKit Authors. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.

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

#define B_SIZE 2048

using namespace c74::min;
namespace max = c74::max;

class DDSPModel
{
public:
    DDSPModel();
    int load(std::string path);
    void perform(double *pitch, double *loudness, double *out_buffer, int buffer_size);

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

void DDSPModel::perform(double *pitch, double *loudness, double *out_buffer, int buffer_size)
{
    torch::NoGradGuard no_grad;
    if (m_model_is_loaded)
    {
        // prepare inputs to the model, explicit type casting necessary
        // model is working with float while Max audio engine is based on double
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        auto pitch_tensor = torch::from_blob(pitch, {1, buffer_size, 1}, options);
        auto loudness_tensor = torch::from_blob(loudness, {1, buffer_size, 1}, options);

        std::vector<torch::jit::IValue> inputs = {pitch_tensor.to(torch::kFloat32), loudness_tensor.to(torch::kFloat32)};

        // run inference
        auto out_tensor = m_scripted_model.forward(inputs).toTensor().to(torch::kFloat64);
        
        // write model output into output buffer
        auto out = out_tensor.contiguous().data_ptr<double>();
        memcpy(out_buffer, out, buffer_size * sizeof(double));
    }
}
class ddsp_audio_decoder_tilde : public object<ddsp_audio_decoder_tilde>, public vector_operator<> {
public:
    MIN_DESCRIPTION     { "DDSP audio decoder~." };
    MIN_TAGS            { "DDSP" };
    MIN_AUTHOR          { "Cycling '74" };
    MIN_RELATED         { "" };

    // execute the ddsp computation in a separate thread
    void thread_perform(double *pitch, double *loudness, double *out_buffer, int buffer_size)
    {
        m_model->perform(pitch, loudness, out_buffer, buffer_size);
    }

    // constructor
    ddsp_audio_decoder_tilde(const atoms& args = {}) {
        // to ensure safety in possible attribute settings
        m_model = new DDSPModel;
            
        // configure inlets and outlets
        auto input_pitch_frequency = std::make_unique<inlet<>>(this, "(signal) pitch frequency");
        auto input_loudness = std::make_unique<inlet<>>(this, "(signal) loudness");
        auto output_ddsp = std::make_unique<outlet<>>(this, "(signal) audio out", "signal");
        
        m_inlets.push_back( std::move(input_pitch_frequency) );
        m_inlets.push_back( std::move(input_loudness) );
        m_outlets.push_back( std::move(output_ddsp) );
    }
                    
    void operator()(audio_bundle input, audio_bundle output) {
        //pitch_buffer[head] = in1; // add sample to the pitch buffer
        //loudness_buffer[head] = in2; // add sample to the loudness buffer
        
        int n = input.frame_count();

        memcpy(pitch_buffer + head, input.samples(0), n * sizeof(double));
        memcpy(loudness_buffer + head, input.samples(1), n * sizeof(double));
        memcpy(output.samples(0), out_buffer + head, output.frame_count() * sizeof(double));

        head += n; // progress with the head

        if (!(head % B_SIZE)) { // if it is B_SIZE or B_SIZE * 2
            if (compute_thread) {
                compute_thread->join();
            }
            
            model_head = ((head + B_SIZE) % (2 * B_SIZE)); // points to the next / previous B_SIZE spaces available
            compute_thread = new std::thread(&ddsp_audio_decoder_tilde::thread_perform, this,
                                            pitch_buffer + model_head,
                                            loudness_buffer + model_head,
                                            out_buffer + model_head,
                                            B_SIZE); // compute the buffers in a separate thread

            head = head % (2 * B_SIZE); // set the head to the next available value
        }
    }
        
    // when object is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "DDSP audio decoder~ loaded" << endl;
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

private:
    // inlets and outlets that will be defined at runtime
    std::vector< std::unique_ptr<inlet<>> > m_inlets;
    std::vector< std::unique_ptr<outlet<>> > m_outlets;

    // controller variables for the model
    DDSPModel *m_model { nullptr };

    // buffers
    double pitch_buffer[2 * B_SIZE];
    double loudness_buffer[2 * B_SIZE];
    double out_buffer[2 * B_SIZE];

    // variable to store the position in the buffer
    int head {0};
    int model_head {0};

    // pointer to run a different thread for the ddsp computation
    std::thread *compute_thread;
    
    // model path
    short m_path {};
    char m_filename[MAX_FILENAME_CHARS] {};
    char m_model_path[MAX_PATH_CHARS] {};
    max::t_fourcc m_type {};
    std::vector<max::t_fourcc> m_types {};
};


MIN_EXTERNAL(ddsp_audio_decoder_tilde);
