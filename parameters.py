import argparse
import json
import itertools
import numpy as np
import random

parser = argparse.ArgumentParser(description='construct parameter files from hyperparams')
parser.add_argument('--dim', type=int)
parser.add_argument('--outputfile', type=str)
parser.add_argument('--model-type', type=str)
parser.add_argument('--initial-hidden', type=int, nargs='+', default=[0])
parser.add_argument('--initial-output', type=int, nargs='+', default=[0])
parser.add_argument('--initial-layers', type=int, nargs='+', default=[0])
parser.add_argument('--phi-hidden', type=int, nargs='+', default=[0])
parser.add_argument('--phi-output', type=int, nargs='+', default=[0])
parser.add_argument('--phi-layers', type=int, nargs='+', default=[0])
parser.add_argument('--rho-hidden', type=int, nargs='+', default=[0])
parser.add_argument('--rho-layers', type=int, nargs='+', default=[0])
parser.add_argument('--num-models', type=int)
parser.add_argument('--from-previous-param', type=str)

args = parser.parse_args()
output_dict = {}

if args.from_previous_param != None:
    f = open(args.from_previous_param)
    previous_parameters = json.load(f)
    f.close()
    print(len(previous_parameters))
    for modelname in previous_parameters:
        for hidden in args.rho_hidden:
            for layers in args.rho_layers:
                rho_hidden = hidden
                rho_layers = layers
                rho = {'hidden': hidden, 'layers': layers}
                phi = previous_parameters[modelname]['phi']
                initial = previous_parameters[modelname]['initial']
                mname = '{prev}-rho-{hidden}-{layers}'.format(prev=modelname, hidden=rho_hidden, layers=rho_layers)
                output_dict[mname]= {'dimension': args.dim, 'initial':initial, 'phi':phi, 'rho':rho}
    print(len(output_dict))
else:
    if args.model_type == 'productnet':
        landmarks = []
        for hidden in args.initial_hidden:
            for output in args.initial_output:
                for layers in args.initial_layers:
                    landmark = (hidden, output, layers, 
                                args.phi_hidden[0], args.phi_output[0], args.phi_layers[0],
                                args.rho_hidden[0], args.rho_layers[0])
                    landmarks.append(landmark)
        product = list(itertools.product(args.initial_hidden, 
                        args.initial_output, 
                        args.initial_layers, 
                        args.phi_hidden, 
                        args.phi_output, 
                        args.phi_layers, 
                        args.rho_hidden,
                        args.rho_layers))
        
        models = random.choices(product, k=args.num_models)
        models = set(models).union(set(landmarks))
        for val in models:
            name = 'init-{}-{}-{}-phi-{}-{}-{}-rho-{}-{}'.format(val[0],
                                                                    val[1],
                                                                    val[2], 
                                                                    val[3],
                                                                    val[4], 
                                                                    val[5], 
                                                                    val[6], 
                                                                    val[7])
            params = {}
            params['dimension'] = args.dim
            params['initial'] = {'hidden': val[0], 'output': val[1], 'layers': val[2]}
            params['phi'] = {'hidden': val[3], 'output': val[4], 'layers': val[5]}
            params['rho'] = {'hidden': val[6], 'layers': val[7]}

            output_dict[name] = params
    elif args.model_type == 'autoencoder-points':
        landmarks = []
        for hidden in args.initial_hidden:
            for output in args.initial_output:
                for layers in args.initial_layers:
                    landmark = (hidden, output, layers, 
                                args.phi_hidden[0], args.phi_output[0], args.phi_layers[0])
                    landmarks.append(landmark)
        product = list(itertools.product(args.initial_hidden, 
                        args.initial_output, 
                        args.initial_layers, 
                        args.phi_hidden, 
                        args.phi_output, 
                        args.phi_layers))
        models = random.choices(product, k=args.num_models)
        models = list(set(models).union(set(landmarks)))
        print("Number of models to train:", len(models))
        for val in models:
            
            name = 'init-{}-{}-{}-phi-{}-{}-{}'.format(val[0],
                                                        val[1],
                                                        val[2], 
                                                        val[3],
                                                        val[4], 
                                                        val[5])
            params = {}
            params['dimension'] = args.dim
            params['initial'] = {'hidden': val[0], 'output': val[1], 'layers': val[2]}
            params['phi'] = {'hidden': val[3], 'output': val[4], 'layers': val[5]}

            output_dict[name] = params
    elif args.model_type == 'autoencoder-images':
        product = args.initial_output

        for val in product:
            
            name = 'esz-{}'.format(val)
            params = {}
            params['dimension'] = args.dim
            params['initial'] = { 'output': val}

            output_dict[name] = params

with open(args.outputfile, 'w') as f_out:
    json.dump(output_dict, f_out, indent=4, sort_keys=True)