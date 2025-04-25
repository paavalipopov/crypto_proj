from Crypto.Cipher import AES, Salsa20, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
import tenseal as ts

import numpy as np

from collections import Counter
from scipy.stats import entropy

def entropy_from_bytes(byte_data):
    counts = Counter(byte_data)
    total = len(byte_data)
    probs = np.array([c / total for c in counts.values()])
    bit_entropy = entropy(probs, base=2)

    int8_array = np.frombuffer(byte_data, dtype=np.int8)
    counts = Counter(int8_array)
    probs = np.array([c / total for c in counts.values()])
    byte_entropy = entropy(probs, base=len(probs))

    return bit_entropy, byte_entropy

def get_data(paths, engine, with_entropy=False):
    # setting up
    if engine == "AES":
        block_size_to_read = 1456
        example_key = get_random_bytes(32)
        cipher = AES.new(example_key, AES.MODE_EAX)
    elif engine == "Salsa":
        block_size_to_read = 1456 - 8 # Salsa20 has an 8-byte nonce
        example_key = get_random_bytes(32)
        # nonce = get_random_bytes(8) # it's better to use a new nonce for each new chunk
        # cipher = Salsa20.new(key=example_key, nonce=nonce)
    elif engine == "RSA":
        block_size_to_read = 210 # RSA block size is proportional to the RSA key size, 
        # and for PKCS1_OAEP it needs to be smaller than vanilla RSA. I haven't found a good formula for this, but 210 < 256 is working
        mykey = RSA.generate(2048)
        pubK = mykey.public_key()
        cipher = PKCS1_OAEP.new(pubK)
    elif engine == "BFV": # homomorphic algorithm
        block_size_to_read = 1456 # here it will be used twice, first to cut the original data bitstream, and second to cut the encrypted BFV vectors (they are really big)
        context = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=4096,
            plain_modulus=1032193
        )
        context.generate_galois_keys()
    else:
        raise NotImplementedError("Oops")
    
    # load the data from each file in paths
    original = []
    encrypted = []
    for path in paths:
        with open(path, 'rb') as f_in:
            while True:
                chunk = f_in.read(block_size_to_read)
                if not chunk:
                    break
                if len(chunk) != block_size_to_read:
                    break
                
                if engine == "BFV":
                    x = np.frombuffer(chunk, dtype=np.uint8)
                    enc_x = ts.bfv_vector(context, x)
                    encrypted_chunk = enc_x.serialize()

                    # Get 10 random byte cuts of size block_size_to_read from encrypted_chunk. Loading them all is too expensive
                    for _ in range(10):
                        start_idx = np.random.randint(0, len(encrypted_chunk) - block_size_to_read)
                        chunk_part = encrypted_chunk[start_idx:start_idx + block_size_to_read]
                        encrypted_array = 2 * np.frombuffer(chunk_part, dtype=np.uint8) / 255 - 0.5
                        if with_entropy:
                            bit_entropy, byte_entropy = entropy_from_bytes(encrypted_array)
                            encrypted_array = np.append(encrypted_array, [bit_entropy, byte_entropy])
                        encrypted.append(encrypted_array)

                    original.append(2*np.frombuffer(chunk, dtype=np.uint8)/255 - 0.5)

                else:
                    if engine == "Salsa":
                        nonce = get_random_bytes(8) # set new nonce
                        cipher = Salsa20.new(key=example_key, nonce=nonce)
                        encrypted_chunk = nonce + cipher.encrypt(chunk)
                    else:
                        encrypted_chunk = cipher.encrypt(chunk)

                    encrypted_array = 2*np.frombuffer(encrypted_chunk, dtype=np.uint8)/255 - 0.5
                    if with_entropy:
                        bit_entropy, byte_entropy = entropy_from_bytes(encrypted_array)
                        encrypted_array = np.append(encrypted_array, [bit_entropy, byte_entropy])

                    # 2*data_sample/255 - 0.5 to map uint8s to [-0.5, 0.5] range
                    original.append(2*np.frombuffer(chunk, dtype=np.uint8)/255 - 0.5)
                    encrypted.append(encrypted_array)
    
    original = np.array(original[:-1])
    encrypted = np.array(encrypted[:-1])

    return original, encrypted


def get_text_data(engine, with_entropy=False):
    paths = ["./data/sample_text.txt", "./data/sample_text_1.txt", "./data/sample_text_2.txt", "./data/sample_text_3.txt", "./data/sample_text_4.txt"]
    return get_data(paths, engine, with_entropy)

def get_voip_data(engine, with_entropy=False):
    paths = ["./data/sample_audio.opus", "./data/sample_audio_1.opus", "./data/sample_audio_2.opus", "./data/sample_audio_3.opus"]
    return get_data(paths, engine, with_entropy)

def get_image_data(engine, with_entropy=False):
    paths = ["./data/sample_image.jpg", "./data/sample_image_1.jpg", "./data/sample_image_2.jpg"]
    return get_data(paths, engine, with_entropy)

def get_little_data(engine, with_entropy=False): # for tests
    paths = ["./data/sample_text.txt"]
    return get_data(paths, engine, with_entropy)