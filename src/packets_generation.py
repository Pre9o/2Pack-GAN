"""
Packet generation module for creating synthetic network packets from GAN output.

This module converts GAN-generated packet representations into valid PCAP files
containing synthetic network traffic (IPv4/UDP/ICMP packets).
"""

import binascii
import os
import datetime
import numpy as np
from keras.models import load_model
from argparse import ArgumentParser
from decoder import decode_packets

UDP_PROTOCOL_ID = '11'
ICMP_PROTOCOL_MAX_LENGTH = 128
LATENT_DIM = 1024  

class PacketGenerationConfig:
    """Manages directory paths for packet generation."""
    
    def __init__(self, results_name=None, base_dir=None):
        """Initialize packet generation configuration.
        
        Args:
            results_name (str, optional): Name of the GAN training results directory
            base_dir (str, optional): Base directory for results. Defaults to ../results
        """
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            base_dir = os.path.join(parent_dir, 'results')
        
        self.results_dir = base_dir
        
        if results_name:
            self.gan_results_dir = os.path.join(base_dir, results_name)
            self.synthetic_packets_dir = os.path.join(self.gan_results_dir, 'synthetic_packets')
            self.models_dir = os.path.join(self.gan_results_dir, 'models')
            
            os.makedirs(self.synthetic_packets_dir, exist_ok=True)
        else:
            self.gan_results_dir = None
            self.synthetic_packets_dir = None
            self.models_dir = None
    
    def get_generator_path(self, epoch):
        """Get path to generator model for specific epoch.
        
        Args:
            epoch (int): Epoch number
            
        Returns:
            str: Path to generator model file
        """
        if not self.models_dir:
            raise ValueError("results_name must be set to get generator path")
        return os.path.join(self.models_dir, f'generator_model{epoch}.keras')
    
    def get_output_pcap_path(self, epoch, num_packets):
        """Get path for output pcap file.
        
        Args:
            epoch (int): Epoch number
            num_packets (int): Number of packets
            
        Returns:
            str: Path to output pcap file
        """
        if not self.synthetic_packets_dir:
            raise ValueError("results_name must be set to get output path")
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(self.synthetic_packets_dir, f'{epoch}_{timestamp}_{num_packets}.pcap')


def pcap_global_header():
    """Returns the PCAP global header (24 bytes).
    
    This header identifies the file as a PCAP capture file and specifies
    the capture format (libpcap 2.4, Ethernet link type).
    
    Returns:
        str: Hexadecimal string representing the PCAP global header
    """
    return ('D4 C3 B2 A1'       # Magic number (little endian)
            '02 00'             # Major version (2)
            '04 00'             # Minor version (4)
            '00 00 00 00'       # GMT to local correction
            '00 00 00 00'       # Timestamp accuracy
            'FF FF 00 00'       # Snapshot length (65535)
            '01 00 00 00')      # Link-layer header type (Ethernet)


def pcap_packet_header():
    """Returns the PCAP packet header template (16 bytes).
    
    Placeholders XX and YY are replaced with actual frame sizes later.
    
    Returns:
        str: Hexadecimal string template for PCAP packet header
    """
    return ('AA 77 9F 47'       # Timestamp seconds
            '90 A2 04 00'       # Timestamp microseconds
            'XX XX XX XX'       # Captured packet length (little endian)
            'YY YY YY YY')      # Original packet length (little endian)


def eth_header():
    """Returns the Ethernet header (14 bytes).
    
    Uses zero MAC addresses and IPv4 EtherType (0x0800).
    
    Returns:
        str: Hexadecimal string representing the Ethernet header
    """
    return ('00 00 00 00 00 00' # Destination MAC (00:00:00:00:00:00)
            '00 00 00 00 00 00' # Source MAC (00:00:00:00:00:00)
            '08 00')            # EtherType (0x0800 = IPv4)


def get_byte_length(hex_string):
    """Calculate byte length of a hexadecimal string.
    
    Args:
        hex_string (str): Hexadecimal string with spaces
        
    Returns:
        integer: Length of the string in bytes
    """
    return int(len(''.join(hex_string.split())) / 2)

def write_bytestring_to_file(bytestring, filename):
    """Write hexadecimal bytestring to binary file.
    
    Args:
        bytestring (str): Hexadecimal string with spaces
        filename (str): Output file path
    """
    hex_bytes = binascii.a2b_hex(''.join(bytestring.split()))
    with open(filename, 'wb') as f:
        f.write(hex_bytes)


def split_into_chunks(string, chunk_size):
    """Split string into chunks of specified size.
    
    Args:
        string (str): Input string
        chunk_size (int): Size of each chunk
        
    Returns:
        list: List of string chunks
    """
    return [string[i:i+chunk_size] for i in range(0, len(string), chunk_size)]


def calculate_ip_checksum(ip_header):
    """Calculate IPv4 header checksum.
    
    Args:
        ip_header (str): IPv4 header as hexadecimal string
        
    Returns:
        int: 16-bit checksum value
    """
    words = split_into_chunks(''.join(ip_header.split()), 4)
    
    checksum = 0
    for word in words:
        checksum += int(word, base=16)
    
    # Add carry bits and apply one's complement
    checksum = (checksum & 0xFFFF) + (checksum >> 16)
    checksum = checksum & 0xFFFF ^ 0xFFFF
    
    return checksum


def calculate_protocol_checksum(protocol_header):
    """Calculate protocol (UDP/TCP) checksum.
    
    Args:
        protocol_header (str): Protocol header as hexadecimal string
        
    Returns:
        int: 16-bit checksum value
    """
    words = split_into_chunks(''.join(protocol_header.split()), 4)
    
    checksum = 0
    for word in words:
        checksum += int(word, base=16)
    
    # Add carry bits and apply one's complement
    checksum = (checksum & 0xFFFF) + (checksum >> 16)
    checksum = checksum & 0xFFFF ^ 0xFFFF
    
    return checksum

def generate_pcap_file(output_path, number_of_packets, ipv4_headers, protocol_headers):
    """Generate a PCAP file from IPv4 and protocol headers.
    
    This function assembles complete network packets by:
    1. Calculating checksums for IP and protocol headers
    2. Adding Ethernet framing
    3. Creating PCAP packet headers with correct lengths
    4. Writing the complete PCAP file
    
    Args:
        output_path (str): Path to the output PCAP file
        number_of_packets (int): Number of packets to generate
        ipv4_headers (list): List of IPv4 header hex strings with placeholders
        protocol_headers (list): List of protocol header hex strings with placeholders
    """
    bytestring = pcap_global_header()
    
    for i in range(number_of_packets):
        # Calculate protocol header length and insert it
        protocol_len = get_byte_length(protocol_headers[i])
        protocol = protocol_headers[i].replace('XXXX', f"{protocol_len:04x}")
        
        # Calculate and insert protocol checksum
        checksum = calculate_protocol_checksum(protocol.replace('YYYY', '00 00'))
        protocol = protocol.replace('YYYY', f"{checksum:04x}")
        
        # Calculate IP total length and insert it
        ip_len = protocol_len + get_byte_length(ipv4_headers[i])
        ip = ipv4_headers[i].replace('XXXX', f"{ip_len:04x}")
        
        # Calculate and insert IP checksum
        checksum = calculate_ip_checksum(ip.replace('YYYY', '00 00'))
        ip = ip.replace('YYYY', f"{checksum:04x}")
        
        # Calculate frame length (IP + protocol + Ethernet)
        frame_len = ip_len + get_byte_length(eth_header())
        
        # Convert to little-endian hex string
        hex_str = f"{frame_len:08x}"
        reverse_hex_str = hex_str[6:8] + hex_str[4:6] + hex_str[2:4] + hex_str[0:2]
        
        # Create packet header with frame length
        pcap_header = pcap_packet_header().replace('XX XX XX XX', reverse_hex_str)
        pcap_header = pcap_header.replace('YY YY YY YY', reverse_hex_str)
        
        # Assemble complete packet
        bytestring += pcap_header + eth_header() + ip + protocol
    
    write_bytestring_to_file(bytestring, output_path)


def generate_packets_by_gan(number_of_packets, epoch, config):
    """Generate synthetic packets using trained GAN model.
    
    Args:
        number_of_packets (int): Number of packets to generate
        epoch (int): Epoch number of the trained model to use
        config (PacketGenerationConfig): Configuration with directory paths
    
    Returns:
        numpy.ndarray: Array of generated packet representations (28x28 matrices)
    
    Raises:
        FileNotFoundError: If generator model file doesn't exist
    """
    generator_path = config.get_generator_path(epoch)
    
    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"Generator model not found: {generator_path}")
    
    print(f"Loading generator from: {generator_path}")
    generator = load_model(generator_path)

    # Generate random noise from latent space
    noise = np.random.normal(0.0, 1.0, (number_of_packets, LATENT_DIM))

    # Generate packet representations
    generated_images = generator.predict(noise, verbose=0)

    # Denormalize from [-1, 1] to [0, 255]
    for i in range(number_of_packets):
        generated_images[i] = (generated_images[i] + 1) * 127.5
        generated_images[i] = generated_images[i].astype(np.uint8)
    
    print(f"{number_of_packets} packets generated by GAN!")
    
    return generated_images


def _prepare_packet_headers(packet_data):
    """Prepare IPv4 and protocol headers from decoded packet data.
    
    Args:
        packet_data (numpy.ndarray): Single packet representation (28x28)
        
    Returns:
        tuple: (ipv4_header, protocol_header) with placeholders for checksums
    """
    raw_ipv4, raw_protocol = decode_packets(packet_data)
    
    # Build IPv4 header with placeholders for length (XXXX) and checksum (YYYY)
    ipv4_header = (raw_ipv4[0:4] + 'XXXX' + raw_ipv4[8:20] + 
                   'YYYY' + raw_ipv4[24:40])
    
    # Determine protocol type and build appropriate header
    is_udp = len(raw_ipv4) > 18 and raw_ipv4[18:20] == UDP_PROTOCOL_ID
    
    if is_udp:
        # UDP: header with length (XXXX) and checksum (YYYY) placeholders
        protocol_header = (raw_protocol[0:8] + 'XXXX' + 'YYYY' + 
                          raw_protocol[16:])
    else:
        # ICMP: simpler structure
        if len(raw_protocol) > ICMP_PROTOCOL_MAX_LENGTH:
            protocol_header = (raw_protocol[0:4] + 'XXXX' + 
                             raw_protocol[8:16] + 
                             raw_protocol[48:ICMP_PROTOCOL_MAX_LENGTH])
        else:
            protocol_header = raw_protocol[0:4] + 'XXXX' + raw_protocol[8:]
    
    return ipv4_header, protocol_header
    

def save_packets_on_training(images, output_path, epoch, examples):
    """Save packets during training (called by GANMonitor callback).
    
    Args:
        images (numpy.ndarray): Generated packet representations from GAN
        output_path (str): Path to save the PCAP file
        epoch (int): Current training epoch
        examples (int): Number of examples to save
    """
    ipv4_headers = []
    protocol_headers = []
    
    for i in range(examples):
        ipv4_header, protocol_header = _prepare_packet_headers(images[i])
        ipv4_headers.append(ipv4_header)
        protocol_headers.append(protocol_header)
    
    generate_pcap_file(output_path, examples, ipv4_headers, protocol_headers)
    print(f"Training PCAP saved: {output_path}")
    
def main():
    """Main function to generate synthetic packets and create PCAP file.
    
    This function:
    1. Loads a trained GAN model
    2. Generates synthetic packet representations
    3. Decodes them into network protocol format
    4. Saves as a valid PCAP file
    """
    arg_parser = ArgumentParser(
        description='Generate synthetic network packets using trained GAN model'
    )

    arg_parser.add_argument('--num_packets', type=int, default=10,help='Number of packets to generate (default: 10)')
    arg_parser.add_argument('--results_name', type=str, required=True,help='Name of GAN training results directory (e.g., dataset_20241210-120000)')
    arg_parser.add_argument('--epoch', type=int, required=True,help='Epoch number of the trained model to use')
    args = arg_parser.parse_args()

    config = PacketGenerationConfig(results_name=args.results_name)
    
    generated_data = generate_packets_by_gan(args.num_packets, args.epoch, config)
    
    ipv4_headers = []
    protocol_headers = []
    
    for i in range(args.num_packets):
        ipv4_header, protocol_header = _prepare_packet_headers(generated_data[i])
        ipv4_headers.append(ipv4_header)
        protocol_headers.append(protocol_header)
        
        if (i + 1) % 10 == 0 or (i + 1) == args.num_packets:
            print(f"Decoded {i + 1}/{args.num_packets} packets")

    output_path = config.get_output_pcap_path(args.epoch, args.num_packets)
    generate_pcap_file(output_path, args.num_packets, ipv4_headers, protocol_headers)
    
    print(f"PCAP file generated successfully!")
    print(f"Output: {output_path}")

if __name__ == '__main__':
    main()