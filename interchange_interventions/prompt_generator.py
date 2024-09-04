import random


PROMPT_TEMPLATE = "Compose a [MOOD] [GENRE] piece, with a [INSTRUMENTS] melody. Use a [TEMPO] tempo."
MOODS = ['emotional', 'passionate', 'groovy', 'energetic', 'happy']
GENRES = ['rock', 'pop', 'electronic', 'jazz', 'classical']
INSTRUMENTS = ['guitar', 'piano', 'trumpet', 'violin', 'flute']
TEMPI = ['fast', 'medium', 'slow', 'slow to medium', 'medium to fast']

type_str_to_list = {
    '[GENRE]': GENRES,
    '[MOOD]': MOODS,
    '[INSTRUMENTS]': INSTRUMENTS,
    "[TEMPO]": TEMPI
}


class PromptGenerator:
    def __init__(self, template=PROMPT_TEMPLATE):
        self.template = template

    def get_prompts_for(self, const_subj, prompt_type, n_total=8, n_per_type=2):
        type_list = ['[GENRE]', '[MOOD]', '[INSTRUMENTS]', '[TEMPO]']
        assert prompt_type in type_list, f"type {prompt_type} not in {type_list}"

        type_list.remove(prompt_type)

        prompts = []
        type_a, type_b, type_c = type_list[0], type_list[1], type_list[2]
        for subj_a in type_str_to_list[type_a][:n_per_type]:
            for subj_b in type_str_to_list[type_b][:n_per_type]:
                for subj_c in type_str_to_list[type_c][:n_per_type]:
                    new_prompt = PROMPT_TEMPLATE.replace(type_a, subj_a).replace(type_b, subj_b).replace(type_c,
                                                                                                         subj_c).replace(
                        prompt_type, const_subj)
                    prompts.append(new_prompt)
        return prompts[:n_total]


def get_nth_prompt(prompt_list, n):
    return prompt_list[n]

if __name__ == '__main__':
    prompt_gen = PromptGenerator()
    prompts = prompt_gen.get_prompts_for('classical', '[GENRE]', n_total=100, n_per_type=1000)
    print("classical")
    print(get_nth_prompt(prompts, 2))
    print(get_nth_prompt(prompts, 44))
    print(get_nth_prompt(prompts, 83))
    print(get_nth_prompt(prompts, 58))
    print(get_nth_prompt(prompts, 41))
    print(get_nth_prompt(prompts, 83))
    print(get_nth_prompt(prompts, 60))
    print(get_nth_prompt(prompts, 88))
    print(get_nth_prompt(prompts, 44))

    print()
    print('electronic')
    prompts = prompt_gen.get_prompts_for('electronic', '[GENRE]', n_total=100, n_per_type=1000)
    print(get_nth_prompt(prompts, 39))
    print(get_nth_prompt(prompts, 64))
    print(get_nth_prompt(prompts, 95))
    print(get_nth_prompt(prompts, 27))
    print(get_nth_prompt(prompts, 66))
    print(get_nth_prompt(prompts, 17))
    print(get_nth_prompt(prompts, 47))
    print(get_nth_prompt(prompts, 84))
    print(get_nth_prompt(prompts, 76))

    print()
    print('guitar')
    prompts = prompt_gen.get_prompts_for('guitar', '[INSTRUMENTS]', n_total=100, n_per_type=1000)
    print(get_nth_prompt(prompts, 58))
    print(get_nth_prompt(prompts, 82))
    print(get_nth_prompt(prompts, 75))
    print(get_nth_prompt(prompts, 13))
    print(get_nth_prompt(prompts, 92))
    print(get_nth_prompt(prompts, 81))
    print(get_nth_prompt(prompts, 47))
    print(get_nth_prompt(prompts, 70))
    print(get_nth_prompt(prompts, 61))

    print()
    print('hiphop')
    prompts = prompt_gen.get_prompts_for('hiphop', '[GENRE]', n_total=100, n_per_type=1000)
    print(get_nth_prompt(prompts, 91))
    print(get_nth_prompt(prompts, 62))
    print(get_nth_prompt(prompts, 13))
    print(get_nth_prompt(prompts, 42))
    print(get_nth_prompt(prompts, 21))
    print(get_nth_prompt(prompts, 87))
    print(get_nth_prompt(prompts, 39))
    print(get_nth_prompt(prompts, 29))
    print(get_nth_prompt(prompts, 64))

    print()
    print('jazz')
    prompts = prompt_gen.get_prompts_for('jazz', '[GENRE]', n_total=100, n_per_type=1000)
    print(get_nth_prompt(prompts, 80))
    print(get_nth_prompt(prompts, 68))
    print(get_nth_prompt(prompts, 90))
    print(get_nth_prompt(prompts, 27))
    print(get_nth_prompt(prompts, 61))
    print(get_nth_prompt(prompts, 67))
    print(get_nth_prompt(prompts, 81))
    print(get_nth_prompt(prompts, 2))
    print(get_nth_prompt(prompts, 97))

    print()
    print('piano')
    prompts = prompt_gen.get_prompts_for('piano', '[INSTRUMENTS]', n_total=100, n_per_type=1000)
    print(get_nth_prompt(prompts, 81))
    print(get_nth_prompt(prompts, 76))
    print(get_nth_prompt(prompts, 28))
    print(get_nth_prompt(prompts, 54))
    print(get_nth_prompt(prompts, 83))
    print(get_nth_prompt(prompts, 31))
    print(get_nth_prompt(prompts, 91))
    print(get_nth_prompt(prompts, 55))
    print(get_nth_prompt(prompts, 84))

    print()
    print('pop')
    prompts = prompt_gen.get_prompts_for('pop', '[GENRE]', n_total=100, n_per_type=1000)
    print(get_nth_prompt(prompts, 65))
    print(get_nth_prompt(prompts, 72))
    print(get_nth_prompt(prompts, 21))
    print(get_nth_prompt(prompts, 48))
    print(get_nth_prompt(prompts, 94))
    print(get_nth_prompt(prompts, 98))
    print(get_nth_prompt(prompts, 84))
    print(get_nth_prompt(prompts, 99))
    print(get_nth_prompt(prompts, 83))

    print()
    print('rock')
    prompts = prompt_gen.get_prompts_for('rock', '[GENRE]', n_total=100, n_per_type=1000)
    print(get_nth_prompt(prompts, 96))
    print(get_nth_prompt(prompts, 73))
    print(get_nth_prompt(prompts, 0))
    print(get_nth_prompt(prompts, 68))
    print(get_nth_prompt(prompts, 67))
    print(get_nth_prompt(prompts, 55))
    print(get_nth_prompt(prompts, 51))
    print(get_nth_prompt(prompts, 89))
    print(get_nth_prompt(prompts, 50))

    print()
    print('trumpet')
    prompts = prompt_gen.get_prompts_for('trumpet', '[INSTRUMENTS]', n_total=100, n_per_type=1000)
    print(get_nth_prompt(prompts, 35))
    print(get_nth_prompt(prompts, 80))
    print(get_nth_prompt(prompts, 73))
    print(get_nth_prompt(prompts, 25))
    print(get_nth_prompt(prompts, 85))
    print(get_nth_prompt(prompts, 48))
    print(get_nth_prompt(prompts, 7))
    print(get_nth_prompt(prompts, 98))
    print(get_nth_prompt(prompts, 24))

    print()
    print('violin')
    prompts = prompt_gen.get_prompts_for('violin', '[INSTRUMENTS]', n_total=100, n_per_type=1000)
    print(get_nth_prompt(prompts, 78))
    print(get_nth_prompt(prompts, 88))
    print(get_nth_prompt(prompts, 19))
    print(get_nth_prompt(prompts, 33))
    print(get_nth_prompt(prompts, 34))
    print(get_nth_prompt(prompts, 33))
    print(get_nth_prompt(prompts, 5))
    print(get_nth_prompt(prompts, 64))
    print(get_nth_prompt(prompts, 60))



