import sys

def main(mode, args):  
    if mode == 'biodenoising_offline':
        import biodenoising_offline 
        biodenoising_offline.main(args)

    elif mode == 'biodenoising_live':
        import biodenoising_live
        biodenoising_live.main(args)
        
if __name__ == "__main__":
    ins = sys.argv
    mode = ins[1]
    args = ins[2:]
    main(mode, args)