use wgpu_native::native::{IntoHandleWithContext, UnwrapId};

#[cfg(feature = "wgpu")]
extern crate wgpu_native;

#[cfg(feature = "tracy")]
mod tracy_impl {
    use tracy_client;
    #[no_mangle]
    pub unsafe extern "C" fn gfxTracyInit() {
        tracy_client::Client::start();
    }
}

pub mod native {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[cfg(all(feature = "wgpu", feature = "vulkan"))]
mod vulkan;

#[macro_export]
macro_rules! expand_chain {
    ($base:expr $(, $chain_type:pat => let $name:ident : $struct_type:ty )*) => {
        $(
            let mut $name: Option<&$struct_type> = None;
        )*
        {
            let mut chain_opt = $base.nextInChain.as_ref();
            while let Some(next_in_chain) = chain_opt {
                match next_in_chain.sType {
                    $(
                        $chain_type => {
                            let ptr = next_in_chain as *const _;
                            let next_in_chain_ptr = ptr as *const $crate::wgpu_native::native::WGPUChainedStruct;
                            assert_eq!(
                                0,
                                next_in_chain_ptr.align_offset(::std::mem::align_of::<$struct_type>()),
                                concat!("Chain structure pointer is not aligned correctly to dereference as ", stringify!($struct_type), ". Correct alignment: {}"),
                                ::std::mem::align_of::<$struct_type>()
                            );
                            $name = Some(&*(next_in_chain_ptr as *const $struct_type));
                        }
                    )*
                    _ => {}
                }
                chain_opt = next_in_chain.next.as_ref();
            }
        }
    };
    ($base:expr $(, $chain_type:pat => let mut $name:ident : $struct_type:ty )*) => {
        $(
            let mut $name: Option<&mut $struct_type> = None;
        )*
        {
            let mut chain_opt = $base.nextInChain.as_mut();
            while let Some(next_in_chain) = chain_opt {
                match next_in_chain.sType {
                    $(
                        $chain_type => {
                            let ptr = next_in_chain as *mut _;
                            let next_in_chain_ptr = ptr as *mut $crate::wgpu_native::native::WGPUChainedStructOut;
                            assert_eq!(
                                0,
                                next_in_chain_ptr.align_offset(::std::mem::align_of::<$struct_type>()),
                                concat!("Chain structure pointer is not aligned correctly to dereference as ", stringify!($struct_type), ". Correct alignment: {}"),
                                ::std::mem::align_of::<$struct_type>()
                            );
                            $name = Some(&mut *(next_in_chain_ptr as *mut $struct_type));
                        }
                    )*
                    _ => {}
                }
                chain_opt = next_in_chain.next.as_mut();
            }
        }
    };
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCreateInstanceEx(
    descriptor: *const wgpu_native::native::WGPUInstanceDescriptor,
) -> wgpu_native::native::WGPUInstance {
    expand_chain!(descriptor.as_ref().unwrap(),
        native::WGPUNativeSTypeEx_InstanceDescriptorVK => let vulkan: native::WGPUInstanceDescriptorVK
    );

    if let Some(vulkan) = &vulkan {
        vulkan::create_instance(vulkan)
    } else {
        wgpu_native::wgpuCreateInstance(Some(&*descriptor))
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceRequestAdapterEx(
    instance: wgpu_native::native::WGPUInstance,
    options: &wgpu_native::native::WGPURequestAdapterOptions,
    callback: wgpu_native::native::WGPURequestAdapterCallback,
    userdata: *mut std::os::raw::c_void,
) {
    expand_chain!(options,
        native::WGPUNativeSTypeEx_RequestAdapterOptionsVK => let vulkan: native::WGPURequestAdapterOptionsVK
    );

    if let Some(vulkan) = &vulkan {
        let adapter = vulkan::create_adapter(instance, &vulkan);
        // TODO: Handle errors
        (callback.unwrap())(
            wgpu_native::native::WGPURequestAdapterStatus_Success,
            adapter,
            std::ptr::null(),
            userdata,
        );
    } else {
        wgpu_native::device::wgpuInstanceRequestAdapter(instance, Some(options), callback, userdata)
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterRequestDeviceEx(
    adapter: wgpu_native::native::WGPUAdapter,
    descriptor: &wgpu_native::native::WGPUDeviceDescriptor,
    callback: wgpu_native::native::WGPURequestDeviceCallback,
    userdata: *mut std::os::raw::c_void,
) {
    expand_chain!(descriptor,
        wgpu_native::native::WGPUSType_DeviceExtras => let extras: wgpu_native::native::WGPUDeviceExtras,
        native::WGPUNativeSTypeEx_DeviceDescriptorVK => let vulkan: native::WGPUDeviceDescriptorVK
    );

    let (adapter_id, context) = adapter.unwrap_handle();

    let (desc, trace_str) = wgpu_native::conv::map_device_descriptor(descriptor, extras);
    let trace_path = trace_str.as_ref().map(std::path::Path::new);

    if let Some(vulkan) = &vulkan {
        let device = vulkan::create_device(&context, adapter_id, &desc, &vulkan);

        match device {
            Some(device) => {
                let (device_id, _err) =
                    context.create_device_from_hal(adapter_id, device, &desc, trace_path, ());
                let handle = device_id.into_handle_with_context(&context);

                (callback.unwrap())(
                    wgpu_native::native::WGPURequestDeviceStatus_Success,
                    handle,
                    std::ptr::null(),
                    userdata,
                );
            }
            None => {
                // TODO: Handle errors
                (callback.unwrap())(
                    wgpu_native::native::WGPURequestDeviceStatus_Error,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    userdata,
                );
            }
        };
    } else {
        wgpu_native::device::wgpuAdapterRequestDevice(adapter, Some(descriptor), callback, userdata);
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceGetPropertiesEx(
    instance: wgpu_native::native::WGPUInstance,
    props: *mut native::WGPUInstanceProperties,
) {
    expand_chain!(props.as_mut().unwrap(),
        native::WGPUNativeSTypeEx_InstancePropertiesVK => let mut vulkan: native::WGPUInstancePropertiesVK
    );

    if let Some(vulkan) = &mut vulkan {
        vulkan::get_instance_properties(instance, vulkan);
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetPropertiesEx(
    device: wgpu_native::native::WGPUDevice,
    props: *mut native::WGPUDeviceProperties,
) {
    expand_chain!(props.as_mut().unwrap(),
        native::WGPUNativeSTypeEx_DevicePropertiesVK => let mut vulkan: native::WGPUDevicePropertiesVK
    );

    if let Some(vulkan) = &mut vulkan {
        vulkan::get_device_properties(device, vulkan);
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCreateExternalTextureView(
    device: wgpu_native::native::WGPUDevice,
    texture_desc: *const wgpu_native::native::WGPUTextureDescriptor,
    view_desc: *const wgpu_native::native::WGPUTextureViewDescriptor,
    descriptor: *const native::WGPUExternalTextureDescriptor,
) -> wgpu_native::native::WGPUTextureView {
    let texture_desc = texture_desc.as_ref().expect("invalid texture descriptor");
    let view_desc = view_desc.as_ref().expect("invalid view descriptor");

    expand_chain!(descriptor.as_ref().unwrap(),
        native::WGPUNativeSTypeEx_ExternalTextureDescriptorVK => let vulkan: native::WGPUTextureViewDescriptorVK
    );

    if let Some(vulkan) = vulkan {
        vulkan::create_external_texture_view(device, texture_desc, view_desc, vulkan)
    } else {
        panic!("can not create external texture view without platform descriptor");
    }
}
