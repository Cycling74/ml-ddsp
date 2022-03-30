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

using namespace c74::min;

class DDSPModel
{
public:
    DDSPModel();
    int load(std::string path);
    at::Tensor perform(float *pitch, float *loudness, float *out_buffer, int buffer_size);

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
        std::cout << e.what() << '\n';
        m_model_is_loaded = 0;
        return 0;
    }
}

at::Tensor DDSPModel::perform(float *pitch, float *loudness, float *out_buffer, int buffer_size)
{
    torch::NoGradGuard no_grad;
    if (m_model_is_loaded)
    {
        auto pitch_tensor = torch::from_blob(pitch, {1, buffer_size, 1});
        auto loudness_tensor = torch::from_blob(loudness, {1, buffer_size, 1});
        auto zero_tensor = torch::zeros({1, buffer_size, 1});

        std::vector<torch::jit::IValue> inputs = {pitch_tensor, loudness_tensor};

        auto out_tensor = m_scripted_model.forward(inputs).toTensor();
        auto out = out_tensor.contiguous().data_ptr<float>();
        memcpy(out_buffer, out, buffer_size * sizeof(float));
        
        return out_tensor;
    }
    return {};
}
class ddsp_decoder_tilde : public object<ddsp_decoder_tilde>, public sample_operator<2,1> {
public:
    MIN_DESCRIPTION	{ "DDSP decoder~." };
    MIN_TAGS		{ "DDSP" };
    MIN_AUTHOR		{ "Cycling '74" };
    MIN_RELATED		{ "" };

    // execute the ddsp computation in a separate thread
    void thread_perform(float *pitch, float *loudness, float *out_buffer, int buffer_size)
    {
        model->perform(pitch, loudness, out_buffer, buffer_size);
    }

    // constructor
    ddsp_decoder_tilde(const atoms& args = {}) {
        // to ensure safety in possible attribute settings
        model = new DDSPModel;
        
        if (args.empty()) {
          error("Please specify the input model path as argument.");
        }
        else {
            cout << "Loading model..." << endl;
            symbol model_path = args[0]; // the first argument specifies the path
            int model_is_loaded = model->load(model_path); // try to load the model
            
            if (model_is_loaded) { // if loaded correctly
            
                // configure inlets and outlets
                auto input_pitch_frequency = std::make_unique<inlet<>>(this, "(signal) pitch frequency");
                auto input_loudness = std::make_unique<inlet<>>(this, "(signal) loudness");
                auto output_ddsp = std::make_unique<outlet<>>(this, "(signal) ddsp out", "signal");
                
                m_inlets.push_back( std::move(input_pitch_frequency) );
                m_inlets.push_back( std::move(input_loudness) );
                m_outlets.push_back( std::move(output_ddsp) );
            
                cout << "Model loaded successfully" << endl;
            }
            else {
                error("Error loading model");
            }
        }
    }
                    
    sample operator()(sample in1, sample in2) {
        pitch_buffer[head] = in1; // add sample to the pitch buffer
        loudness_buffer[head] = in2; // add sample to the loudness buffer
        
        head++; // progress with the head
        
        if (!(head % B_SIZE)) // if it is B_SIZE or B_SIZE * 2
        {
            model_head = ((head + B_SIZE) % (2 * B_SIZE)); // points to the next / previous B_SIZE spaces available
            compute_thread = new std::thread(&ddsp_decoder_tilde::thread_perform, this,
                                            pitch_buffer + model_head,
                                            loudness_buffer + model_head,
                                            out_buffer + model_head,
                                            B_SIZE); // compute the buffers in a separate thread

            head = head % (2 * B_SIZE); // set the head to the next available value
        }
                    
        return out_buffer[head]; // outputs the output buffer
    }
        
    // when object is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "DDSP decoder~ loaded" << endl;
            return {};
        }
    };

private:
    // inlets and outlets that will be defined at runtime
    std::vector< std::unique_ptr<inlet<>> > m_inlets;
    std::vector< std::unique_ptr<outlet<>> > m_outlets;

    // controller variables for the model
    DDSPModel *model { nullptr };

    // buffers
    float pitch_buffer[2 * B_SIZE];
    float loudness_buffer[2 * B_SIZE];
    float out_buffer[2 * B_SIZE];

    // variable to store the position in the buffer
    int head {0};
    int model_head {0};

    // pointer to run a different thread for the ddsp computation
    std::thread *compute_thread;
};


MIN_EXTERNAL(ddsp_decoder_tilde);
