Ч£
Ц+ш*
:
Add
x"T
y"T
z"T"
Ttype:
2	
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
Є
AsString

input"T

output"
Ttype:
2		
"
	precisionint€€€€€€€€€"

scientificbool( "
shortestbool( "
widthint€€€€€€€€€"
fillstring 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
S
	Bucketize

input"T

output"
Ttype:
2	"

boundarieslist(float)
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
Ц
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
Р
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
Ј
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
z
SparseSegmentMean	
data"T
indices"Tidx
segment_ids
output"T"
Ttype:
2"
Tidxtype0:
2	
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
А
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.13.12
b'unknown'Э±

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
П
global_step
VariableV2*
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container *
shape: 
≤
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
f
SepalLengthPlaceholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
e

SepalWidthPlaceholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
f
PetalLengthPlaceholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
e

PetalWidthPlaceholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
“
Зdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"   
   *w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
≈
Жdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
«
Иdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *Ыи°>*w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
ч
Сdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalЗdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
_output_shapes

:
*

seed *
T0*w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0*
seed2 *
dtype0
ќ
Еdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulСdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalИdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0*
_output_shapes

:

Љ
Бdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normalAddЕdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulЖdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0*
_output_shapes

:

—
ddnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0*
	container *
shape
:

™
kdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/AssignAssignddnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0Бdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0
э
idnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/readIdentityddnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0*
T0*w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0*
_output_shapes

:

Р
Ednn/input_from_feature_columns/input_layer/PetalLength/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
б
Adnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims
ExpandDimsPetalLengthEdnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
љ
<dnn/input_from_feature_columns/input_layer/PetalLength/ShapeShapeAdnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims*
T0*
out_type0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
М
Ddnn/input_from_feature_columns/input_layer/PetalLength/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/PetalLength/ShapeJdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stackLdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
И
Fdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ф
Ddnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/PetalLength/strided_sliceFdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Т
>dnn/input_from_feature_columns/input_layer/PetalLength/ReshapeReshapeAdnn/input_from_feature_columns/input_layer/PetalLength/ExpandDimsDdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
П
Ddnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dimConst*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
ё
@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims
ExpandDims
PetalWidthDdnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
ї
;dnn/input_from_feature_columns/input_layer/PetalWidth/ShapeShape@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims*
out_type0*
_output_shapes
:*
T0
У
Idnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Х
Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Х
Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
З
Cdnn/input_from_feature_columns/input_layer/PetalWidth/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/PetalWidth/ShapeIdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stackKdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
З
Ednn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
С
Cdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/PetalWidth/strided_sliceEdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1*
_output_shapes
:*
T0*

axis *
N
П
=dnn/input_from_feature_columns/input_layer/PetalWidth/ReshapeReshape@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDimsCdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Р
Ednn/input_from_feature_columns/input_layer/SepalLength/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
б
Adnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims
ExpandDimsSepalLengthEdnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€*

Tdim0
љ
<dnn/input_from_feature_columns/input_layer/SepalLength/ShapeShapeAdnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims*
T0*
out_type0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Ц
Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
М
Ddnn/input_from_feature_columns/input_layer/SepalLength/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/SepalLength/ShapeJdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stackLdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
И
Fdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ф
Ddnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/SepalLength/strided_sliceFdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Т
>dnn/input_from_feature_columns/input_layer/SepalLength/ReshapeReshapeAdnn/input_from_feature_columns/input_layer/SepalLength/ExpandDimsDdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
ч
Udnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Bucketize	BucketizeAdnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims*

boundaries
"Ќћћ=  А?  »B*
T0*'
_output_shapes
:€€€€€€€€€
ж
Qdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ShapeShapeUdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Bucketize*
T0*
out_type0*
_output_shapes
:
©
_dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ђ
adnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ђ
adnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
х
Ydnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_sliceStridedSliceQdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Shape_dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice/stackadnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice/stack_1adnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Щ
Wdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
Щ
Wdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ш
Qdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/rangeRangeWdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range/startYdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_sliceWdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
Ь
Zdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
—
Vdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ExpandDims
ExpandDimsQdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/rangeZdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€*

Tdim0
Ђ
Zdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Tile/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:
–
Pdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/TileTileVdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ExpandDimsZdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Tile/multiples*'
_output_shapes
:€€€€€€€€€*

Tmultiples0*
T0
ђ
Ydnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
«
Sdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ReshapeReshapePdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/TileYdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
Ы
Ydnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
Ы
Ydnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range_1/limitConst*
value	B :*
dtype0*
_output_shapes
: 
Ы
Ydnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Х
Sdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range_1RangeYdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range_1/startYdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range_1/limitYdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range_1/delta*
_output_shapes
:*

Tidx0
щ
\dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Tile_1/multiplesPackYdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice*
T0*

axis *
N*
_output_shapes
:
Ќ
Rdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Tile_1TileSdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/range_1\dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Tile_1/multiples*#
_output_shapes
:€€€€€€€€€*

Tmultiples0*
T0
Ѓ
[dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape_1/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
–
Udnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape_1ReshapeUdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Bucketize[dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape_1/shape*
Tshape0*#
_output_shapes
:€€€€€€€€€*
T0
У
Qdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/mul/xConst*
dtype0*
_output_shapes
: *
value	B :
Ђ
Odnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/mulMulQdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/mul/xRdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Tile_1*#
_output_shapes
:€€€€€€€€€*
T0
ђ
Odnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/addAddUdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape_1Odnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/mul*
T0*#
_output_shapes
:€€€€€€€€€
…
Qdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/stackPackSdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ReshapeRdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Tile_1*'
_output_shapes
:€€€€€€€€€*
T0*

axis *
N
Ђ
Zdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/transpose/permConst*
dtype0*
_output_shapes
:*
valueB"       
–
Udnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/transpose	TransposeQdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/stackZdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/transpose/perm*'
_output_shapes
:€€€€€€€€€*
Tperm0*
T0
Г
Sdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ToInt64CastUdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/transpose*'
_output_shapes
:€€€€€€€€€*

DstT0	*

SrcT0*
Truncate( 
Ч
Udnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/stack_1/1Const*
value	B :*
dtype0*
_output_shapes
: 
«
Sdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/stack_1PackYdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_sliceUdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/stack_1/1*
T0*

axis *
N*
_output_shapes
:
ц
Udnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ToInt64_1CastSdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/stack_1*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0	
Ћ
Аdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0
…
dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
Р
zdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SliceSliceUdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ToInt64_1Аdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice/begindnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
ƒ
zdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ј
ydnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/ProdProdzdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slicezdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Const*
T0	*
_output_shapes
: *
	keep_dims( *

Tidx0
»
Еdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
≈
Вdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
±
}dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2GatherV2Udnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ToInt64_1Еdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2/indicesВdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
Ј
{dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Cast/xPackydnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Prod}dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2*

axis *
N*
_output_shapes
:*
T0	
м
Вdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseReshapeSparseReshapeSdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ToInt64Udnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ToInt64_1{dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
Ч
Лdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseReshape/IdentityIdentityOdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/add*#
_output_shapes
:€€€€€€€€€*
T0
∆
Гdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B : *
dtype0
‘
Бdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GreaterEqualGreaterEqualЛdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseReshape/IdentityГdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0
є
zdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/WhereWhereБdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€*
T0

÷
Вdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Reshape/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
ƒ
|dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/ReshapeReshapezdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/WhereВdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:€€€€€€€€€
«
Дdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
к
dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2_1GatherV2Вdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseReshape|dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/ReshapeДdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2_1/axis*
Tindices0	*
Tparams0	*'
_output_shapes
:€€€€€€€€€*
Taxis0
«
Дdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
п
dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2_2GatherV2Лdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseReshape/Identity|dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/ReshapeДdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2_2/axis*#
_output_shapes
:€€€€€€€€€*
Taxis0*
Tindices0	*
Tparams0
µ
}dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/IdentityIdentityДdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
—
Оdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0*
_output_shapes
: *
value	B : 
•
Ьdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2_1dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/GatherV2_2}dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/IdentityОdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0
т
†dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
ф
Ґdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
ф
Ґdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
÷
Ъdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceЬdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows†dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackҐdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Ґdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*#
_output_shapes
:€€€€€€€€€
Д
Сdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/CastCastЪdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*
Truncate( *#
_output_shapes
:€€€€€€€€€*

DstT0
Л
Уdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/UniqueUniqueЮdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
out_idx0*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ё
Ґdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Ю
Эdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2idnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/readУdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/UniqueҐdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:€€€€€€€€€
*
Taxis0*
Tindices0*
Tparams0*w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0
Е
¶dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityЭdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€

µ
Мdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparseSparseSegmentMean¶dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityХdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/Unique:1Сdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse/Cast*

Tidx0*
T0*'
_output_shapes
:€€€€€€€€€

÷
Дdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
с
~dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Reshape_1ReshapeЮdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Дdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0
*
Tshape0
«
zdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/ShapeShapeМdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse*
T0*
out_type0*
_output_shapes
:
”
Иdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
’
Кdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
’
Кdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∆
Вdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/strided_sliceStridedSlicezdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/ShapeИdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/strided_slice/stackКdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/strided_slice/stack_1Кdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
Њ
|dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
њ
zdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/stackPack|dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/stack/0Вdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/strided_slice*
T0*

axis *
N*
_output_shapes
:
 
ydnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/TileTile~dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Reshape_1zdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ќ
dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/zeros_like	ZerosLikeМdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€
*
T0
ї
tdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weightsSelectydnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Tilednn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/zeros_likeМdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€

Ю
{dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Cast_1CastUdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/ToInt64_1*

SrcT0	*
Truncate( *
_output_shapes
:*

DstT0
Ќ
Вdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
ћ
Бdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
љ
|dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_1Slice{dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Cast_1Вdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_1/beginБdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
∞
|dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Shape_1Shapetdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights*
T0*
out_type0*
_output_shapes
:
Ќ
Вdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
’
Бdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_2/sizeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
Њ
|dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_2Slice|dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Shape_1Вdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_2/beginБdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
√
Аdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ј
{dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/concatConcatV2|dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_1|dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Slice_2Аdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
Љ
~dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Reshape_2Reshapetdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights{dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/concat*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

С
Sdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Shape_1Shape~dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Reshape_2*
T0*
out_type0*
_output_shapes
:
Ђ
adnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
≠
cdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
≠
cdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
€
[dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice_1StridedSliceSdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Shape_1adnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice_1/stackcdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice_1/stack_1cdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
Я
]dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape_2/shape/1Const*
dtype0*
_output_shapes
: *
value	B :

ў
[dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape_2/shapePack[dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/strided_slice_1]dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape_2/shape/1*
T0*

axis *
N*
_output_shapes
:
э
Udnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape_2Reshape~dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/SepalLength_bucketized_embedding_weights/Reshape_2[dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

П
Ddnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
ё
@dnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims
ExpandDims
SepalWidthDdnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
ї
;dnn/input_from_feature_columns/input_layer/SepalWidth/ShapeShape@dnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims*
T0*
out_type0*
_output_shapes
:
У
Idnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Х
Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Х
Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
З
Cdnn/input_from_feature_columns/input_layer/SepalWidth/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/SepalWidth/ShapeIdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stackKdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
З
Ednn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
С
Cdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/SepalWidth/strided_sliceEdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
П
=dnn/input_from_feature_columns/input_layer/SepalWidth/ReshapeReshape@dnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDimsCdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
С
1dnn/input_from_feature_columns/input_layer/concatConcatV2>dnn/input_from_feature_columns/input_layer/PetalLength/Reshape=dnn/input_from_feature_columns/input_layer/PetalWidth/Reshape>dnn/input_from_feature_columns/input_layer/SepalLength/ReshapeUdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/Reshape_2=dnn/input_from_feature_columns/input_layer/SepalWidth/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:€€€€€€€€€
≈
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"   
   *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:
Ј
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *   њ*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
Ј
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *   ?*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
Ю
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:

Ъ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
ђ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:

Ю
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:
*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
я
dnn/hiddenlayer_0/kernel/part_0VarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!dnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container *
shape
:

П
@dnn/hiddenlayer_0/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
Ў
&dnn/hiddenlayer_0/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0
«
3dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:

Ѓ
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*
valueB
*    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:

’
dnn/hiddenlayer_0/bias/part_0VarHandleOp*
	container *
shape:
*
dtype0*
_output_shapes
: *.
shared_namednn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
Л
>dnn/hiddenlayer_0/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
: 
«
$dnn/hiddenlayer_0/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
dtype0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
љ
1dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:

З
'dnn/hiddenlayer_0/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:

v
dnn/hiddenlayer_0/kernelIdentity'dnn/hiddenlayer_0/kernel/ReadVariableOp*
_output_shapes

:
*
T0
«
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
transpose_b( 

%dnn/hiddenlayer_0/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:

n
dnn/hiddenlayer_0/biasIdentity%dnn/hiddenlayer_0/bias/ReadVariableOp*
T0*
_output_shapes
:

Я
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€

k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*'
_output_shapes
:€€€€€€€€€
*
T0
g
dnn/zero_fraction/SizeSizednn/hiddenlayer_0/Relu*
T0*
out_type0	*
_output_shapes
: 
c
dnn/zero_fraction/LessEqual/yConst*
valueB	 R€€€€*
dtype0	*
_output_shapes
: 
А
dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
Д
dnn/zero_fraction/cond/SwitchSwitchdnn/zero_fraction/LessEqualdnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
m
dnn/zero_fraction/cond/switch_tIdentitydnn/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
k
dnn/zero_fraction/cond/switch_fIdentitydnn/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
h
dnn/zero_fraction/cond/pred_idIdentitydnn/zero_fraction/LessEqual*
_output_shapes
: *
T0

С
*dnn/zero_fraction/cond/count_nonzero/zerosConst ^dnn/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
ѕ
-dnn/zero_fraction/cond/count_nonzero/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1*dnn/zero_fraction/cond/count_nonzero/zeros*
T0*'
_output_shapes
:€€€€€€€€€

ж
4dnn/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*:
_output_shapes(
&:€€€€€€€€€
:€€€€€€€€€
*
T0*)
_class
loc:@dnn/hiddenlayer_0/Relu
±
)dnn/zero_fraction/cond/count_nonzero/CastCast-dnn/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *'
_output_shapes
:€€€€€€€€€
*

DstT0
Э
*dnn/zero_fraction/cond/count_nonzero/ConstConst ^dnn/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
ќ
2dnn/zero_fraction/cond/count_nonzero/nonzero_countSum)dnn/zero_fraction/cond/count_nonzero/Cast*dnn/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Ч
dnn/zero_fraction/cond/CastCast2dnn/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0*
Truncate( 
У
,dnn/zero_fraction/cond/count_nonzero_1/zerosConst ^dnn/zero_fraction/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
”
/dnn/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch,dnn/zero_fraction/cond/count_nonzero_1/zeros*'
_output_shapes
:€€€€€€€€€
*
T0
и
6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*)
_class
loc:@dnn/hiddenlayer_0/Relu*:
_output_shapes(
&:€€€€€€€€€
:€€€€€€€€€
*
T0
µ
+dnn/zero_fraction/cond/count_nonzero_1/CastCast/dnn/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *'
_output_shapes
:€€€€€€€€€
*

DstT0	
Я
,dnn/zero_fraction/cond/count_nonzero_1/ConstConst ^dnn/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
‘
4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countSum+dnn/zero_fraction/cond/count_nonzero_1/Cast,dnn/zero_fraction/cond/count_nonzero_1/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
§
dnn/zero_fraction/cond/MergeMerge4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countdnn/zero_fraction/cond/Cast*
_output_shapes
: : *
T0	*
N
Ж
(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
Ы
)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
Л
+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
∞
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
†
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
dtype0*
_output_shapes
: *>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values
ѓ
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_0/activation/tagConst*
_output_shapes
: *1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0
У
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
≈
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"
   
   *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
Ј
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *М7њ*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
Ј
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *М7?*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0
Ю
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:

*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
seed2 
Ъ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
ђ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:


Ю
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:

*
T0
я
dnn/hiddenlayer_1/kernel/part_0VarHandleOp*
shape
:

*
dtype0*
_output_shapes
: *0
shared_name!dnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container 
П
@dnn/hiddenlayer_1/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
Ў
&dnn/hiddenlayer_1/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0
«
3dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:


Ѓ
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
valueB
*    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:

’
dnn/hiddenlayer_1/bias/part_0VarHandleOp*
dtype0*
_output_shapes
: *.
shared_namednn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container *
shape:

Л
>dnn/hiddenlayer_1/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
«
$dnn/hiddenlayer_1/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0
љ
1dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:

З
'dnn/hiddenlayer_1/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:


v
dnn/hiddenlayer_1/kernelIdentity'dnn/hiddenlayer_1/kernel/ReadVariableOp*
_output_shapes

:

*
T0
ђ
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
transpose_b( 

%dnn/hiddenlayer_1/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:

n
dnn/hiddenlayer_1/biasIdentity%dnn/hiddenlayer_1/bias/ReadVariableOp*
T0*
_output_shapes
:

Я
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€

k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*'
_output_shapes
:€€€€€€€€€
*
T0
i
dnn/zero_fraction_1/SizeSizednn/hiddenlayer_1/Relu*
T0*
out_type0	*
_output_shapes
: 
e
dnn/zero_fraction_1/LessEqual/yConst*
valueB	 R€€€€*
dtype0	*
_output_shapes
: 
Ж
dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 
К
dnn/zero_fraction_1/cond/SwitchSwitchdnn/zero_fraction_1/LessEqualdnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_1/cond/switch_tIdentity!dnn/zero_fraction_1/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_1/cond/switch_fIdentitydnn/zero_fraction_1/cond/Switch*
_output_shapes
: *
T0

l
 dnn/zero_fraction_1/cond/pred_idIdentitydnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: 
Х
,dnn/zero_fraction_1/cond/count_nonzero/zerosConst"^dnn/zero_fraction_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
’
/dnn/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_1/cond/count_nonzero/zeros*'
_output_shapes
:€€€€€€€€€
*
T0
к
6dnn/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*:
_output_shapes(
&:€€€€€€€€€
:€€€€€€€€€
*
T0*)
_class
loc:@dnn/hiddenlayer_1/Relu
µ
+dnn/zero_fraction_1/cond/count_nonzero/CastCast/dnn/zero_fraction_1/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *'
_output_shapes
:€€€€€€€€€
*

DstT0
°
,dnn/zero_fraction_1/cond/count_nonzero/ConstConst"^dnn/zero_fraction_1/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
‘
4dnn/zero_fraction_1/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_1/cond/count_nonzero/Cast,dnn/zero_fraction_1/cond/count_nonzero/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Ы
dnn/zero_fraction_1/cond/CastCast4dnn/zero_fraction_1/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
Ч
.dnn/zero_fraction_1/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_1/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
ў
1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_1/cond/count_nonzero_1/zeros*'
_output_shapes
:€€€€€€€€€
*
T0
м
8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_1/Relu*:
_output_shapes(
&:€€€€€€€€€
:€€€€€€€€€

є
-dnn/zero_fraction_1/cond/count_nonzero_1/CastCast1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *'
_output_shapes
:€€€€€€€€€
*

DstT0	
£
.dnn/zero_fraction_1/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_1/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Џ
6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_1/cond/count_nonzero_1/Cast.dnn/zero_fraction_1/cond/count_nonzero_1/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
™
dnn/zero_fraction_1/cond/MergeMerge6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_1/cond/Cast*
T0	*
N*
_output_shapes
: : 
М
*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Sizednn/zero_fraction_1/cond/Merge*
_output_shapes
: *
T0	
Я
+dnn/zero_fraction_1/counts_to_fraction/CastCast*dnn/zero_fraction_1/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
П
-dnn/zero_fraction_1/counts_to_fraction/Cast_1Castdnn/zero_fraction_1/Size*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
ґ
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
_output_shapes
: *
T0
†
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
±
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
У
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
Ј
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
:
©
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *тк-њ*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
©
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *тк-?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
Й
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:
*

seed 
ю
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*+
_class!
loc:@dnn/logits/kernel/part_0
Р
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:

В
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:
*
T0*+
_class!
loc:@dnn/logits/kernel/part_0
 
dnn/logits/kernel/part_0VarHandleOp*+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape
:
*
dtype0*
_output_shapes
: *)
shared_namednn/logits/kernel/part_0
Б
9dnn/logits/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel/part_0*
_output_shapes
: 
Љ
dnn/logits/kernel/part_0/AssignAssignVariableOpdnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
≤
,dnn/logits/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:

†
(dnn/logits/bias/part_0/Initializer/zerosConst*
valueB*    *)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
ј
dnn/logits/bias/part_0VarHandleOp*
dtype0*
_output_shapes
: *'
shared_namednn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
	container *
shape:
}
7dnn/logits/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias/part_0*
_output_shapes
: 
Ђ
dnn/logits/bias/part_0/AssignAssignVariableOpdnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*)
_class
loc:@dnn/logits/bias/part_0*
dtype0
®
*dnn/logits/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
y
 dnn/logits/kernel/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:

h
dnn/logits/kernelIdentity dnn/logits/kernel/ReadVariableOp*
T0*
_output_shapes

:

Ю
dnn/logits/MatMulMatMuldnn/hiddenlayer_1/Reludnn/logits/kernel*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( 
q
dnn/logits/bias/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
`
dnn/logits/biasIdentitydnn/logits/bias/ReadVariableOp*
T0*
_output_shapes
:
К
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
e
dnn/zero_fraction_2/SizeSizednn/logits/BiasAdd*
_output_shapes
: *
T0*
out_type0	
e
dnn/zero_fraction_2/LessEqual/yConst*
valueB	 R€€€€*
dtype0	*
_output_shapes
: 
Ж
dnn/zero_fraction_2/LessEqual	LessEqualdnn/zero_fraction_2/Sizednn/zero_fraction_2/LessEqual/y*
T0	*
_output_shapes
: 
К
dnn/zero_fraction_2/cond/SwitchSwitchdnn/zero_fraction_2/LessEqualdnn/zero_fraction_2/LessEqual*
_output_shapes
: : *
T0

q
!dnn/zero_fraction_2/cond/switch_tIdentity!dnn/zero_fraction_2/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_2/cond/switch_fIdentitydnn/zero_fraction_2/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_2/cond/pred_idIdentitydnn/zero_fraction_2/LessEqual*
_output_shapes
: *
T0

Х
,dnn/zero_fraction_2/cond/count_nonzero/zerosConst"^dnn/zero_fraction_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
’
/dnn/zero_fraction_2/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_2/cond/count_nonzero/zeros*'
_output_shapes
:€€€€€€€€€*
T0
в
6dnn/zero_fraction_2/cond/count_nonzero/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_2/cond/pred_id*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*
T0
µ
+dnn/zero_fraction_2/cond/count_nonzero/CastCast/dnn/zero_fraction_2/cond/count_nonzero/NotEqual*
Truncate( *'
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0

°
,dnn/zero_fraction_2/cond/count_nonzero/ConstConst"^dnn/zero_fraction_2/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
‘
4dnn/zero_fraction_2/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_2/cond/count_nonzero/Cast,dnn/zero_fraction_2/cond/count_nonzero/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Ы
dnn/zero_fraction_2/cond/CastCast4dnn/zero_fraction_2/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
Ч
.dnn/zero_fraction_2/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_2/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
ў
1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_2/cond/count_nonzero_1/zeros*'
_output_shapes
:€€€€€€€€€*
T0
д
8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_2/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€
є
-dnn/zero_fraction_2/cond/count_nonzero_1/CastCast1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *'
_output_shapes
:€€€€€€€€€*

DstT0	
£
.dnn/zero_fraction_2/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_2/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Џ
6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_2/cond/count_nonzero_1/Cast.dnn/zero_fraction_2/cond/count_nonzero_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0	
™
dnn/zero_fraction_2/cond/MergeMerge6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_2/cond/Cast*
N*
_output_shapes
: : *
T0	
М
*dnn/zero_fraction_2/counts_to_fraction/subSubdnn/zero_fraction_2/Sizednn/zero_fraction_2/cond/Merge*
T0	*
_output_shapes
: 
Я
+dnn/zero_fraction_2/counts_to_fraction/CastCast*dnn/zero_fraction_2/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
П
-dnn/zero_fraction_2/counts_to_fraction/Cast_1Castdnn/zero_fraction_2/Size*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
ґ
.dnn/zero_fraction_2/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_2/counts_to_fraction/Cast-dnn/zero_fraction_2/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
y
dnn/zero_fraction_2/fractionIdentity.dnn/zero_fraction_2/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Т
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
£
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/fraction*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst*
dtype0*
_output_shapes
: **
value!B Bdnn/dnn/logits/activation
Б
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
g
dnn/head/logits/ShapeShapednn/logits/BiasAdd*
T0*
out_type0*
_output_shapes
:
k
)dnn/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
[
Sdnn/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
s
(dnn/head/predictions/class_ids/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
≥
dnn/head/predictions/class_idsArgMaxdnn/logits/BiasAdd(dnn/head/predictions/class_ids/dimension*
output_type0	*#
_output_shapes
:€€€€€€€€€*

Tidx0*
T0
n
#dnn/head/predictions/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
∞
dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/class_ids#dnn/head/predictions/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:€€€€€€€€€
Ё
 dnn/head/predictions/str_classesAsStringdnn/head/predictions/ExpandDims*

scientific( *
width€€€€€€€€€*'
_output_shapes
:€€€€€€€€€*
	precision€€€€€€€€€*
shortest( *
T0	*

fill 
s
"dnn/head/predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
p
dnn/head/ShapeShape"dnn/head/predictions/probabilities*
out_type0*
_output_shapes
:*
T0
f
dnn/head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
dnn/head/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
dnn/head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¶
dnn/head/strided_sliceStridedSlicednn/head/Shapednn/head/strided_slice/stackdnn/head/strided_slice/stack_1dnn/head/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
V
dnn/head/range/startConst*
_output_shapes
: *
value	B : *
dtype0
V
dnn/head/range/limitConst*
value	B :*
dtype0*
_output_shapes
: 
V
dnn/head/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Б
dnn/head/rangeRangednn/head/range/startdnn/head/range/limitdnn/head/range/delta*
_output_shapes
:*

Tidx0
∞
dnn/head/AsStringAsStringdnn/head/range*
T0*

fill *

scientific( *
width€€€€€€€€€*
_output_shapes
:*
	precision€€€€€€€€€*
shortest( 
Y
dnn/head/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
В
dnn/head/ExpandDims
ExpandDimsdnn/head/AsStringdnn/head/ExpandDims/dim*
T0*
_output_shapes

:*

Tdim0
[
dnn/head/Tile/multiples/1Const*
_output_shapes
: *
value	B :*
dtype0
М
dnn/head/Tile/multiplesPackdnn/head/strided_slicednn/head/Tile/multiples/1*
T0*

axis *
N*
_output_shapes
:
З
dnn/head/TileTilednn/head/ExpandDimsdnn/head/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
shape: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
r
save/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:

X
save/IdentityIdentitysave/Read/ReadVariableOp*
T0*
_output_shapes
:

^
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes
:

z
save/Read_1/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:

`
save/Identity_2Identitysave/Read_1/ReadVariableOp*
T0*
_output_shapes

:

d
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
_output_shapes

:
*
T0
t
save/Read_2/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:

\
save/Identity_4Identitysave/Read_2/ReadVariableOp*
T0*
_output_shapes
:

`
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
_output_shapes
:
*
T0
z
save/Read_3/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:


`
save/Identity_6Identitysave/Read_3/ReadVariableOp*
_output_shapes

:

*
T0
d
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
T0*
_output_shapes

:


m
save/Read_4/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
\
save/Identity_8Identitysave/Read_4/ReadVariableOp*
_output_shapes
:*
T0
`
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
_output_shapes
:*
T0
s
save/Read_5/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:

a
save/Identity_10Identitysave/Read_5/ReadVariableOp*
T0*
_output_shapes

:

f
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes

:

Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_e076e813952242d8a227bb2590b58c7f/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Џ
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
valuevBtB]dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weightsBglobal_step*
dtype0
Г
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*$
valueBB4 10 0,4:0,10B 
ь
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesidnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/readglobal_step"/device:CPU:0*
dtypes
2	
†
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0
Р
save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Г
save/Read_6/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:

l
save/Identity_12Identitysave/Read_6/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:

b
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
T0*
_output_shapes
:

Й
save/Read_7/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:

p
save/Identity_14Identitysave/Read_7/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:

f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
_output_shapes

:
*
T0
Г
save/Read_8/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:

l
save/Identity_16Identitysave/Read_8/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:

b
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
T0*
_output_shapes
:

Й
save/Read_9/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:


p
save/Identity_18Identitysave/Read_9/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:


f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
T0*
_output_shapes

:


}
save/Read_10/ReadVariableOpReadVariableOpdnn/logits/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_20Identitysave/Read_10/ReadVariableOp"/device:CPU:0*
_output_shapes
:*
T0
b
save/Identity_21Identitysave/Identity_20"/device:CPU:0*
_output_shapes
:*
T0
Г
save/Read_11/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:

q
save/Identity_22Identitysave/Read_11/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:

f
save/Identity_23Identitysave/Identity_22"/device:CPU:0*
T0*
_output_shapes

:

ы
save/SaveV2_1/tensor_namesConst"/device:CPU:0*Э
valueУBРBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernel*
dtype0*
_output_shapes
:
Њ
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*]
valueTBRB10 0,10B14 10 0,14:0,10B10 0,10B10 10 0,10:0,10B3 0,3B10 3 0,10:0,3*
dtype0*
_output_shapes
:
ь
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_13save/Identity_15save/Identity_17save/Identity_19save/Identity_21save/Identity_23"/device:CPU:0*
dtypes

2
®
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
а
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*
N*
_output_shapes
:*
T0*

axis 
М
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
®
save/Identity_24Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
Ё
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valuevBtB]dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weightsBglobal_step*
dtype0*
_output_shapes
:
Ж
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*$
valueBB4 10 0,4:0,10B *
dtype0*
_output_shapes
:
™
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*"
_output_shapes
:
:*
dtypes
2	
÷
save/AssignAssignddnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0save/RestoreV2*w
_classm
kiloc:@dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
†
save/Assign_1Assignglobal_stepsave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
8
save/restore_shardNoOp^save/Assign^save/Assign_1
ю
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*Э
valueУBРBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernel*
dtype0*
_output_shapes
:
Ѕ
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*]
valueTBRB10 0,10B14 10 0,14:0,10B10 0,10B10 10 0,10:0,10B3 0,3B10 3 0,10:0,3*
dtype0*
_output_shapes
:
÷
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*D
_output_shapes2
0:
:
:
:

::
*
dtypes

2
b
save/Identity_25Identitysave/RestoreV2_1"/device:CPU:0*
T0*
_output_shapes
:

v
save/AssignVariableOpAssignVariableOpdnn/hiddenlayer_0/bias/part_0save/Identity_25"/device:CPU:0*
dtype0
h
save/Identity_26Identitysave/RestoreV2_1:1"/device:CPU:0*
T0*
_output_shapes

:

z
save/AssignVariableOp_1AssignVariableOpdnn/hiddenlayer_0/kernel/part_0save/Identity_26"/device:CPU:0*
dtype0
d
save/Identity_27Identitysave/RestoreV2_1:2"/device:CPU:0*
_output_shapes
:
*
T0
x
save/AssignVariableOp_2AssignVariableOpdnn/hiddenlayer_1/bias/part_0save/Identity_27"/device:CPU:0*
dtype0
h
save/Identity_28Identitysave/RestoreV2_1:3"/device:CPU:0*
T0*
_output_shapes

:


z
save/AssignVariableOp_3AssignVariableOpdnn/hiddenlayer_1/kernel/part_0save/Identity_28"/device:CPU:0*
dtype0
d
save/Identity_29Identitysave/RestoreV2_1:4"/device:CPU:0*
T0*
_output_shapes
:
q
save/AssignVariableOp_4AssignVariableOpdnn/logits/bias/part_0save/Identity_29"/device:CPU:0*
dtype0
h
save/Identity_30Identitysave/RestoreV2_1:5"/device:CPU:0*
T0*
_output_shapes

:

s
save/AssignVariableOp_5AssignVariableOpdnn/logits/kernel/part_0save/Identity_30"/device:CPU:0*
dtype0
≈
save/restore_shard_1NoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"?
save/Const:0save/Identity_24:0save/restore_all (5 @F8"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"я 
cond_contextќ Ћ 
ђ
 dnn/zero_fraction/cond/cond_text dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_t:0 *ј
dnn/hiddenlayer_0/Relu:0
dnn/zero_fraction/cond/Cast:0
+dnn/zero_fraction/cond/count_nonzero/Cast:0
,dnn/zero_fraction/cond/count_nonzero/Const:0
6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
/dnn/zero_fraction/cond/count_nonzero/NotEqual:0
4dnn/zero_fraction/cond/count_nonzero/nonzero_count:0
,dnn/zero_fraction/cond/count_nonzero/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_t:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0R
dnn/hiddenlayer_0/Relu:06dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ы
"dnn/zero_fraction/cond/cond_text_1 dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_f:0*ѓ
dnn/hiddenlayer_0/Relu:0
-dnn/zero_fraction/cond/count_nonzero_1/Cast:0
.dnn/zero_fraction/cond/count_nonzero_1/Const:0
8dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
1dnn/zero_fraction/cond/count_nonzero_1/NotEqual:0
6dnn/zero_fraction/cond/count_nonzero_1/nonzero_count:0
.dnn/zero_fraction/cond/count_nonzero_1/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_f:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0T
dnn/hiddenlayer_0/Relu:08dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
 
"dnn/zero_fraction_1/cond/cond_text"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_t:0 *Ў
dnn/hiddenlayer_1/Relu:0
dnn/zero_fraction_1/cond/Cast:0
-dnn/zero_fraction_1/cond/count_nonzero/Cast:0
.dnn/zero_fraction_1/cond/count_nonzero/Const:0
8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_1/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_1/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_1/cond/count_nonzero/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_t:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0T
dnn/hiddenlayer_1/Relu:08dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
Ј
$dnn/zero_fraction_1/cond/cond_text_1"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_f:0*≈
dnn/hiddenlayer_1/Relu:0
/dnn/zero_fraction_1/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_1/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_1/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_f:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0V
dnn/hiddenlayer_1/Relu:0:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
¬
"dnn/zero_fraction_2/cond/cond_text"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_t:0 *–
dnn/logits/BiasAdd:0
dnn/zero_fraction_2/cond/Cast:0
-dnn/zero_fraction_2/cond/count_nonzero/Cast:0
.dnn/zero_fraction_2/cond/count_nonzero/Const:0
8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_2/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_2/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_2/cond/count_nonzero/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_t:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0P
dnn/logits/BiasAdd:08dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
ѓ
$dnn/zero_fraction_2/cond/cond_text_1"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_f:0*љ
dnn/logits/BiasAdd:0
/dnn/zero_fraction_2/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_2/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_2/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_f:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0R
dnn/logits/BiasAdd:0:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0"%
saved_model_main_op


group_deps"з
trainable_variablesѕћ
Ј
fdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0:0kdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Assignkdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/read:0"k
]dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights
  "
2Гdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
м
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernel
  "
(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
÷
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias
 "
(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
м
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel

  "

(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
÷
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias
 "
(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
…
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel
  "
(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
≥
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08"В
	summariesф
с
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"Ј
	variables©¶
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
Ј
fdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0:0kdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Assignkdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/read:0"k
]dnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights
  "
2Гdnn/input_from_feature_columns/input_layer/SepalLength_bucketized_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
м
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernel
  "
(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
÷
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias
 "
(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
м
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel

  "

(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
÷
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias
 "
(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
…
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel
  "
(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
≥
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08*ъ
predictо
-

PetalWidth
PetalWidth:0€€€€€€€€€
-

SepalWidth
SepalWidth:0€€€€€€€€€
/
SepalLength 
SepalLength:0€€€€€€€€€
/
PetalLength 
PetalLength:0€€€€€€€€€E
	class_ids8
!dnn/head/predictions/ExpandDims:0	€€€€€€€€€L
probabilities;
$dnn/head/predictions/probabilities:0€€€€€€€€€D
classes9
"dnn/head/predictions/str_classes:0€€€€€€€€€5
logits+
dnn/logits/BiasAdd:0€€€€€€€€€tensorflow/serving/predict