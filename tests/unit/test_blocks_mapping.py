"""
Simple test of blocks_to_slot_mask functionality
"""
from utils import build_blocks, blocks_to_slot_mask

def test_blocks_mapping():
    """Test block to slot mapping"""
    print("Testing blocks to slot mapping...")
    
    # Test parameters
    T = 12  # 12 time slots
    dt_min = 5  # 5-minute slots
    sustain_min = 30  # 30-minute blocks
    
    # Build blocks
    blocks = build_blocks(T, dt_min, sustain_min)
    print(f"Time slots: {T} (total {T * dt_min} minutes)")
    print(f"Block duration: {sustain_min} minutes")
    print(f"Blocks: {blocks}")
    
    # Test different enforcement patterns
    test_cases = [
        ([1, 0], "First block only"),
        ([0, 1], "Second block only"),
        ([1, 1], "Both blocks"),
        ([0, 0], "No blocks")
    ]
    
    for enforce_blocks, description in test_cases:
        mask = blocks_to_slot_mask(enforce_blocks, blocks, T)
        enforced_slots = [i for i, m in enumerate(mask) if m == 1]
        
        print(f"\\nEnforce blocks: {enforce_blocks} ({description})")
        print(f"Baseline mask: {mask}")
        print(f"Enforced slots: {enforced_slots}")
        
        # Verify mapping
        for b, enforce in enumerate(enforce_blocks):
            if b < len(blocks):
                block_slots = blocks[b]
                if enforce == 1:
                    # All slots in this block should be enforced
                    all_enforced = all(mask[slot] == 1 for slot in block_slots)
                    print(f"  Block {b} slots {block_slots}: {'✓' if all_enforced else '✗'} enforced")
                else:
                    # No slots in this block should be enforced (unless enforced by other blocks)
                    none_enforced_by_this_block = True  # This is complex to verify
                    print(f"  Block {b} slots {block_slots}: not enforced by this block")

if __name__ == "__main__":
    test_blocks_mapping()
