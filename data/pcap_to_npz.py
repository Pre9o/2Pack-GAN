import numpy as np
import os
import pyshark
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

pcap_dir = os.path.join(parent_dir, 'pcaps')
npz_dir = os.path.join(current_dir, 'npz')

if not os.path.exists(f"{npz_dir}"):
    os.makedirs(f"{npz_dir}")
    

def packet_useful_data(packet, start, end):
    """
    Returns the useful data of a packet

    Args:
        packet (string): Packet

    Returns:
        string: Useful data of the packet
    """    
    return packet[start:end]

def packet_means(packet):
    """
    Returns the means of the bytes of a packet
    
    Args:
        packet (string): Packet
        
    Returns:
        list: List of means of the bytes of a packet
    """
    packet_list = []
    packet_with_mean = []
    
    for i in range(0, len(packet), 2):
        packet_list.append(packet[i:i+2])
        
    for i in range(0, 14):
        packet_list.append('00')
                
    for i in range(0, len(packet_list)):
        packet_with_mean.append(packet_list[i][:1] + '8')
        packet_with_mean.append(packet_list[i][1:] + '8')
        
    return packet_with_mean

def duplicate_and_map_bytes(byte_digits, n=28, d=2):
    """
    Duplicates and maps the bytes of a packet
    
    Args:
        byte_digits (list): List of bytes
        n (integer): Size of the matrix
        d (integer): Size of the submatrix
        
    Returns:
        numpy.ndarray: Matrix of bytes
    """
    
    # Initialize the n x n matrix with zeros of object type
    matrix = np.zeros((n, n), dtype=object)

    i, j = 0, 0
    for byte_digit in byte_digits:
        # Fill the d x d submatrix with the entered byte value (as string)
        value = int(byte_digit, 16)  # Convert the hexadecimal string to integer
        matrix[i:i+d, j:j+d] = value

        # Update the indexes
        j += d
        if j >= n:
            j = 0
            i += d
            if i >= n:
                break

    return matrix

def which_protocol(packet):
    """
    Returns the protocol of a packet
    
    Args:
        packet (string): Packet
        
    Returns:
        string: Protocol of the packet
    """
    if packet[18:20] == '11' and (packet[46:48] == '35' or packet[52:54] == '35'):
        return 'DNS'
    elif packet[18:20] == '01':
        return 'ICMP'
    elif packet[18:20] == '11':
        return 'UDP'
    else:
        return 'Other'
    

def process_pcap(pcap_name, packets_limit, useful_data_start, useful_data_end, desired_protocol, npz_file):
    """
    Processes a pcap file and generates a dataset

    Args:
        pcap_name (string): Name of the pcap file
        packets_limit (integer): Number of packets to be processed
        useful_data_start (integer): Start of the useful data
        useful_data_end (integer): End of the useful data
        desired_protocol (string): Protocol to be processed
        npz_file (string): Name of the npz file
    """
    pcap_path = os.path.join(pcap_dir, pcap_name)
    
    dataset = {"x_train": [], "y_train": []}
    
    pcap = pyshark.FileCapture(pcap_path, use_json=True, include_raw=True)
        
    index = 0
    
    for pkt in pcap:
        print(f"Packet {index + 1}")
        
        packet = pkt.frame_raw.value        
        packet = packet_useful_data(packet, useful_data_start, useful_data_end)
        protocol = which_protocol(packet)
        
        if protocol == desired_protocol:
            
            packet = packet_means(packet)
            packet = duplicate_and_map_bytes(packet)
            
            dataset["x_train"].append(packet.tolist())
            dataset["y_train"].append(1)
                
            index += 1
        if index == packets_limit:
            break
            
    np.savez(os.path.join(npz_dir, npz_file), **dataset)
    

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Process pcap file and generate dataset.")
    parser.add_argument("--pcap_name", type=str, default='default_pcap', help="Name of the pcap file")
    parser.add_argument("--packets_limit", type=int, default=20000, help="Number of packets to be processed")
    parser.add_argument("--useful_data_start", type=int, default=28, help="Start of the useful data, default is 28 (without Ethernet header)")
    parser.add_argument("--useful_data_end", type=int, default=300, help="End of the useful data, default is 300")
    parser.add_argument("--protocol", type=str, default='DNS', help="Protocol to be processed")
    parser.add_argument("--npz_file", type=str, default='default_npz', help="Name of the npz file")
    
    args = parser.parse_args()
    
    pcap_name = args.pcap_name
    if not pcap_name.endswith('.pcap'):
        pcap_name += '.pcap'
    
    packets_limit = args.packets_limit
    useful_data_start = args.useful_data_start
    useful_data_end = args.useful_data_end
    desired_protocol = args.protocol
    
    npz_file = args.npz_file
    if not npz_file.endswith('.npz'):
        npz_file += '.npz'
    
    process_pcap(pcap_name, packets_limit, useful_data_start, useful_data_end, desired_protocol, npz_file)
        
if __name__ == "__main__":
    main()