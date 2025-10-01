import gemmi
import numpy as np

def crop_operations(operations, fractional_box):
    min_extent = np.array(list(fractional_box.minimum))
    max_extent = np.array(list(fractional_box.maximum))

    offset = max_extent - min_extent

    selected_operations = []
    for operation in operations:
        min_extent_operation = np.array(operation.apply_to_xyz(min_extent))
        max_extent_operation = np.array(operation.apply_to_xyz(max_extent))
        if (min_extent_operation >= (min_extent - offset)).all() and (max_extent_operation <= (max_extent + offset)).all():
            selected_operations.append(operation)
    return selected_operations

def apply_operation_power(operations):
    operations = operations + [gemmi.Op("x+1,y,z"), gemmi.Op("x,y+1,z"), gemmi.Op("x,y,z+1")]
    operations = [op for operation in operations for op in [operation, operation.inverse()]]
    new_operations = operations[:]
    for operation in operations:
        for other_operation in operations:
            new_operations.append(operation.combine(other_operation))
    new_operations = list(set(new_operations))
    operations = new_operations[:]
    for operation in operations:
        for other_operation in operations:
            new_operations.append(operation.combine(other_operation))
    operations = list(set(new_operations))
    return operations

def get_R_T_from_operations(space_group, unit_cell, fractional_box, include_identity=False):
    operations = apply_operation_power(list(space_group.operations()))
    operations = crop_operations(operations, fractional_box)
    operations = [op for op in operations if (op != gemmi.Op("x,y,z") or include_identity)]
    R_T = []
    for operation in operations:
        transform = unit_cell.op_as_transform(operation)
        R, T = np.array(transform.mat), np.array(list(transform.vec))
        R_T.append((R, T))
    return R_T

def get_pdb_symmetries_R_T(pdb: gemmi.Structure, include_identity=False):
    cell = pdb.cell
    space_groups = gemmi.SpaceGroup(pdb.spacegroup_hm)
    fractional_box = pdb.calculate_fractional_box()
    R_Ts = get_R_T_from_operations(space_groups, cell, fractional_box, include_identity=include_identity)
    return R_Ts