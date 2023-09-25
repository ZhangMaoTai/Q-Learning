import re
import collections


class BaselineModel(object):
    def __init__(self, vocab_path):

        full_dictionary_location = vocab_path
        self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()

        self.current_dictionary = []

    def guess(self, word, guessed_letters):  # word input example: "_ p p _ e "
        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word[::2].replace("_", ".")

        # find length of passed word
        len_word = len(clean_word)

        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []

        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue

            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(clean_word, dict_word):
                new_dictionary.append(dict_word)

        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary

        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)

        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()

        guess_letter = '!'

        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter, instance_count in sorted_letter_count:
            if letter not in guessed_letters:
                guess_letter = letter
                break

        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter, instance_count in sorted_letter_count:
                if letter not in guessed_letters:
                    guess_letter = letter
                    break

        return guess_letter

    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location, "r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary


if __name__ == "__main__":
    baseline = BaselineModel("../words_240000.txt")
    print(baseline.guess("_ _ _ _ _ _ ", []))
    print(baseline.guess("_ _ _ _ _ _ ", ["e"]))
    print(baseline.guess("_ _ _ _ _ _ ", ["e", "i"]))
    print(baseline.guess("g _ m ", list("gm")))
    print(baseline.guess("a _ _ l e ", list("aple")))

