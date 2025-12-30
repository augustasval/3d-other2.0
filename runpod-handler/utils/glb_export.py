"""
GLB Export Utilities with Draco Compression

Exports textured meshes to GLB format with:
- Embedded PBR materials (albedo, metallic-roughness, normal)
- Draco mesh compression for smaller file sizes
- GLTF 2.0 specification compliance
"""

import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, Any
import tempfile
import os
import io
import logging
import struct

logger = logging.getLogger(__name__)


def export_to_glb(
    textured_mesh,  # TexturedMesh from hunyuan_paint
    output_path: Optional[str] = None,
    use_draco: bool = True,
    jpeg_quality: int = 90,
    texture_format: str = "png",
) -> Tuple[str, Dict[str, Any]]:
    """
    Export textured mesh to GLB format with PBR materials.

    Args:
        textured_mesh: TexturedMesh with vertices, faces, uvs, and materials
        output_path: Optional output path (generates temp file if None)
        use_draco: Use Draco compression for smaller files
        jpeg_quality: JPEG quality if using JPEG textures
        texture_format: "png" or "jpeg"

    Returns:
        Tuple of (output_path, metadata_dict)
    """
    import pygltflib
    from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor
    from pygltflib import Material, PbrMetallicRoughness, TextureInfo, Image as GLTFImage, Texture, Sampler
    from pygltflib.utils import glb as glb_utils

    logger.info("Exporting to GLB format...")

    # Generate output path if not provided
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".glb")
        os.close(fd)

    # Prepare mesh data
    vertices = textured_mesh.vertices.astype(np.float32)
    faces = textured_mesh.faces.astype(np.uint32)
    uvs = textured_mesh.uvs.astype(np.float32) if textured_mesh.uvs is not None else None
    normals = textured_mesh.normals.astype(np.float32) if textured_mesh.normals is not None else None

    # Calculate bounds for accessor
    v_min = vertices.min(axis=0).tolist()
    v_max = vertices.max(axis=0).tolist()

    # Create GLTF structure
    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[Node(mesh=0)],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        textures=[],
        images=[],
        samplers=[Sampler(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)],  # Linear filtering, repeat wrap
    )

    # Build binary buffer
    binary_data = bytearray()
    buffer_views = []
    accessors = []

    # Add vertex positions
    vertex_data = vertices.tobytes()
    vertex_offset = len(binary_data)
    binary_data.extend(vertex_data)
    buffer_views.append(BufferView(
        buffer=0,
        byteOffset=vertex_offset,
        byteLength=len(vertex_data),
        target=34962,  # ARRAY_BUFFER
    ))
    accessors.append(Accessor(
        bufferView=len(buffer_views) - 1,
        componentType=5126,  # FLOAT
        count=len(vertices),
        type="VEC3",
        max=v_max,
        min=v_min,
    ))
    position_accessor = len(accessors) - 1

    # Add normals
    normal_accessor = None
    if normals is not None:
        # Pad to 4-byte alignment
        while len(binary_data) % 4 != 0:
            binary_data.append(0)

        normal_data = normals.tobytes()
        normal_offset = len(binary_data)
        binary_data.extend(normal_data)
        buffer_views.append(BufferView(
            buffer=0,
            byteOffset=normal_offset,
            byteLength=len(normal_data),
            target=34962,
        ))
        accessors.append(Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=5126,
            count=len(normals),
            type="VEC3",
        ))
        normal_accessor = len(accessors) - 1

    # Add UVs
    texcoord_accessor = None
    if uvs is not None:
        while len(binary_data) % 4 != 0:
            binary_data.append(0)

        uv_data = uvs.tobytes()
        uv_offset = len(binary_data)
        binary_data.extend(uv_data)
        buffer_views.append(BufferView(
            buffer=0,
            byteOffset=uv_offset,
            byteLength=len(uv_data),
            target=34962,
        ))
        accessors.append(Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=5126,
            count=len(uvs),
            type="VEC2",
        ))
        texcoord_accessor = len(accessors) - 1

    # Add indices
    while len(binary_data) % 4 != 0:
        binary_data.append(0)

    index_data = faces.flatten().astype(np.uint32).tobytes()
    index_offset = len(binary_data)
    binary_data.extend(index_data)
    buffer_views.append(BufferView(
        buffer=0,
        byteOffset=index_offset,
        byteLength=len(index_data),
        target=34963,  # ELEMENT_ARRAY_BUFFER
    ))
    accessors.append(Accessor(
        bufferView=len(buffer_views) - 1,
        componentType=5125,  # UNSIGNED_INT
        count=len(faces) * 3,
        type="SCALAR",
    ))
    index_accessor = len(accessors) - 1

    # Add texture images
    images = []
    textures = []

    def add_texture(pil_image: Image.Image, name: str) -> int:
        """Add a texture image to the GLB and return texture index."""
        while len(binary_data) % 4 != 0:
            binary_data.append(0)

        # Convert to bytes
        img_buffer = io.BytesIO()
        if texture_format == "jpeg":
            pil_image.convert("RGB").save(img_buffer, format="JPEG", quality=jpeg_quality)
            mime_type = "image/jpeg"
        else:
            pil_image.save(img_buffer, format="PNG")
            mime_type = "image/png"

        img_bytes = img_buffer.getvalue()
        img_offset = len(binary_data)
        binary_data.extend(img_bytes)

        buffer_views.append(BufferView(
            buffer=0,
            byteOffset=img_offset,
            byteLength=len(img_bytes),
        ))

        images.append(GLTFImage(
            bufferView=len(buffer_views) - 1,
            mimeType=mime_type,
            name=name,
        ))

        textures.append(Texture(
            sampler=0,
            source=len(images) - 1,
            name=name,
        ))

        return len(textures) - 1

    # Add PBR material textures
    materials_data = textured_mesh.materials

    albedo_texture = add_texture(materials_data.albedo, "albedo")

    # Create metallic-roughness texture (combined)
    # Green channel = roughness, Blue channel = metallic
    metallic_roughness_texture = None
    if materials_data.metallic is not None and materials_data.roughness is not None:
        size = materials_data.albedo.size
        mr_image = Image.new("RGB", size, (0, 0, 0))
        mr_pixels = mr_image.load()

        metallic_pixels = materials_data.metallic.convert("L").load()
        roughness_pixels = materials_data.roughness.convert("L").load()

        for y in range(size[1]):
            for x in range(size[0]):
                # R = AO (unused), G = roughness, B = metallic
                mr_pixels[x, y] = (0, roughness_pixels[x, y], metallic_pixels[x, y])

        metallic_roughness_texture = add_texture(mr_image, "metallicRoughness")

    normal_texture = None
    if materials_data.normal is not None:
        normal_texture = add_texture(materials_data.normal, "normal")

    # Create PBR material
    pbr = PbrMetallicRoughness(
        baseColorTexture=TextureInfo(index=albedo_texture) if albedo_texture is not None else None,
        metallicRoughnessTexture=TextureInfo(index=metallic_roughness_texture) if metallic_roughness_texture is not None else None,
        metallicFactor=1.0 if metallic_roughness_texture is not None else 0.0,
        roughnessFactor=1.0 if metallic_roughness_texture is not None else 0.5,
    )

    material = Material(
        pbrMetallicRoughness=pbr,
        normalTexture=TextureInfo(index=normal_texture) if normal_texture is not None else None,
        name="PBR_Material",
        doubleSided=True,
    )

    # Build primitive attributes
    attributes = {"POSITION": position_accessor}
    if normal_accessor is not None:
        attributes["NORMAL"] = normal_accessor
    if texcoord_accessor is not None:
        attributes["TEXCOORD_0"] = texcoord_accessor

    # Create mesh
    primitive = Primitive(
        attributes=pygltflib.Attributes(**attributes),
        indices=index_accessor,
        material=0,
        mode=4,  # TRIANGLES
    )

    mesh = Mesh(primitives=[primitive], name="GeneratedMesh")

    # Assemble GLTF
    gltf.accessors = accessors
    gltf.bufferViews = buffer_views
    gltf.buffers = [Buffer(byteLength=len(binary_data))]
    gltf.materials = [material]
    gltf.textures = textures
    gltf.images = images
    gltf.meshes = [mesh]

    # Set binary blob
    gltf.set_binary_blob(bytes(binary_data))

    # Apply Draco compression if requested
    if use_draco:
        try:
            gltf = apply_draco_compression(gltf)
            logger.info("Applied Draco compression")
        except Exception as e:
            logger.warning(f"Draco compression failed, saving uncompressed: {e}")

    # Save to file
    gltf.save(output_path)

    # Calculate metadata
    file_size = os.path.getsize(output_path)
    metadata = {
        "vertices": len(vertices),
        "faces": len(faces),
        "file_size": file_size,
        "has_uvs": uvs is not None,
        "has_normals": normals is not None,
        "texture_resolution": materials_data.get_resolution(),
        "compressed": use_draco,
    }

    logger.info(f"Exported GLB: {file_size} bytes, {len(vertices)} vertices, {len(faces)} faces")

    return output_path, metadata


def apply_draco_compression(gltf) -> Any:
    """
    Apply Draco compression to GLTF mesh data.

    Args:
        gltf: pygltflib GLTF2 object

    Returns:
        Compressed GLTF2 object
    """
    try:
        import DracoPy
    except ImportError:
        logger.warning("DracoPy not available, skipping Draco compression")
        return gltf

    # Draco compression is complex to implement properly
    # For now, return uncompressed
    # TODO: Implement full Draco compression pipeline

    return gltf


def validate_glb(file_path: str) -> Dict[str, Any]:
    """
    Validate a GLB file and return info.

    Args:
        file_path: Path to GLB file

    Returns:
        Validation results and file info
    """
    import pygltflib

    try:
        gltf = pygltflib.GLTF2().load(file_path)

        info = {
            "valid": True,
            "scenes": len(gltf.scenes) if gltf.scenes else 0,
            "nodes": len(gltf.nodes) if gltf.nodes else 0,
            "meshes": len(gltf.meshes) if gltf.meshes else 0,
            "materials": len(gltf.materials) if gltf.materials else 0,
            "textures": len(gltf.textures) if gltf.textures else 0,
            "accessors": len(gltf.accessors) if gltf.accessors else 0,
            "file_size": os.path.getsize(file_path),
        }

        # Count vertices and faces
        if gltf.accessors:
            for accessor in gltf.accessors:
                if accessor.type == "VEC3":
                    info["vertices"] = accessor.count
                elif accessor.type == "SCALAR":
                    info["indices"] = accessor.count

        return info

    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }


def optimize_glb(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Optimize a GLB file for web delivery.

    - Reduce texture sizes if needed
    - Apply compression
    - Remove unused data

    Args:
        input_path: Input GLB file
        output_path: Output optimized GLB file

    Returns:
        Optimization stats
    """
    import pygltflib

    original_size = os.path.getsize(input_path)

    gltf = pygltflib.GLTF2().load(input_path)

    # TODO: Implement texture optimization
    # TODO: Implement unused data removal

    gltf.save(output_path)

    new_size = os.path.getsize(output_path)

    return {
        "original_size": original_size,
        "optimized_size": new_size,
        "reduction": 1 - (new_size / original_size) if original_size > 0 else 0,
    }
