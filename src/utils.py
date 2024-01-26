def word_wrap(string, n_chars=72):
    """
    Helper function to wrap text to a certain number of characters.
    """
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)
    

def extract_main_topics(text):
    """
    Extracts main topics from a given text.
    Main topics are lines that are formatted as '= topic ='.
    
    :param text: A string containing the text with topics.
    :return: A list of main topics.
    """
    main_topics = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('= ') and line.endswith(' =') and line.count('=') == 2:
            topic = line.strip('= ').strip()
            main_topics.append(topic)
    return main_topics

def extract_all_topics(text):
    """
    Extracts all topics from a given text, excluding lines that are only '='.
    Topics are lines that are formatted with '=' at both ends and contain alphanumeric characters.
    
    :param text: A string containing the text with topics.
    :return: A list of all topics, without any '='.
    """
    topics = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('=') and line.endswith('='):
            # Remove all '=' characters and join the remaining parts
            topic = ''.join(part.strip() for part in line.split('=') if part.strip())
            # Exclude topics that don't contain alphanumeric characters
            if any(char.isalnum() for char in topic):
                topics.append(topic)
    return topics

def extract_chars_around_string(text, search_string, before, after):
    """
    Search for a given string in the text and output a specified number of characters 
    before and after that string.

    :param text: The text to search in.
    :param search_string: The string to search for.
    :param before: The number of characters to output before the found string.
    :param after: The number of characters to output after the found string.
    :return: A substring containing the specified number of characters before and 
             after the search string.
    """
    # Find the index of the given string
    start_index = text.find(search_string)
    
    # Check if the string was found
    if start_index == -1:
        return "String not found"

    # Calculate the starting and ending indices
    start = max(0, start_index - before)  # Ensure we don't go beyond the start of the text
    end = min(len(text), start_index + len(search_string) + after)  # Ensure we don't go beyond the end of the text

    # Extract and return the substring
    return text[start:end]

def count_chars_in_topics(original_text):
    ''' 
    Counts the number of characters in each topic.
    '''
    lines = original_text.split('\n')
    current_topic = None
    char_count = 0
    topic_char_counts = {}

    for line in lines:
        level = line.count('=')
        if level > 0:
            if current_topic is not None:
                topic_char_counts[current_topic] = char_count
            current_topic = line.strip(' =')
            char_count = 0
        else:
            char_count += len(line)

    # For the last topic
    if current_topic is not None:
        topic_char_counts[current_topic] = char_count

    return topic_char_counts

def modify_topics(original_text):
    """
    Modifies the topics in a given text to be hierarchical.
    For example, if the original text has the following topics:
    = topic 1 =
    == topic 2 ==

    The modified text will have the following topics:
    = topic 1 =
    == topic 1-topic 2 ==
    """

    # Split the text into lines
    lines = original_text.split("\n")

    # This will store the current topic at each level
    current_topics = {}

    # Result list to store the modified lines
    modified_lines = []

    for line in lines:

        # Count the number of '=' on one side of the line
        level = line.count("=") // 2

        # If level is 0, it's random text or an empty line
        if level == 0:
            modified_lines.append(line)
            continue

        # Extract the topic
        topic = line.strip(" =")

        # Update the current topic at this level
        current_topics[level] = topic

        # Remove higher level topics if we go back to a lower level
        for l in list(current_topics.keys()):
            if l > level:
                del current_topics[l]

        # Build the new topic line
        new_topic = "-".join(
            current_topics[l] for l in sorted(current_topics) if l <= level
        )
        modified_line = " " + "= " * level + new_topic + " =" * level

        # Add to the modified lines
        modified_lines.append(modified_line)

    # Join the modified lines back into a single string
    return "\n".join(modified_lines)