import argparse
import tqdm
from pathlib import Path

from transformers import pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, help='path to directory with pretrained pipeline')
    parser.add_argument('--data', type=str, help='path to directory inference data. Each text should be in a separate line')
    parser.add_argument('--output-file', type=str, default='inference_output.txt', help='path to the file where results will be stored.')

    args = parser.parse_args()

    model_pipeline = pipeline('text-classification', args.model_dir)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(args.data) as f_in:
        with open(str(output_file), 'w') as f_out:
            for text in tqdm.tqdm(f_in):
                text = text.strip()
                result = model_pipeline([text])[0]
                f_out.write(f'{result["label"]} {result["score"]:0.4f}\n')
