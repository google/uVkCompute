// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "uvkc/vulkan/driver.h"

#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "uvkc/base/status.h"
#include "uvkc/base/target_platform.h"
#include "uvkc/vulkan/dynamic_symbols.h"
#include "uvkc/vulkan/status_util.h"

namespace uvkc {
namespace vulkan {

namespace {

VkApplicationInfo GetDefaultApplicationInfo(const char *app_name) {
  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = nullptr;
  app_info.pApplicationName = app_name;
  app_info.applicationVersion = 0;
  app_info.pEngineName = "uVkCompute";
  app_info.engineVersion = 0;
#if defined(UVKC_PLATFORM_ANDROID)
  app_info.apiVersion = VK_API_VERSION_1_1;
#else
  app_info.apiVersion = VK_API_VERSION_1_2;
#endif
  return app_info;
}

// Selects a queue family with the required |queue_flags| in |physical_device|
// If found, writes the |valid_timestamp_bits| and returns the queue family
// index.
absl::StatusOr<uint32_t> SelectQueueFamily(VkPhysicalDevice physical_device,
                                           VkQueueFlags queue_flags,
                                           uint32_t *valid_timestamp_bits,
                                           const DynamicSymbols &symbols) {
  uint32_t count;
  symbols.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count,
                                                   nullptr);

  std::vector<VkQueueFamilyProperties> queue_families(count);
  symbols.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count,
                                                   queue_families.data());

  for (int index = 0; index < count; ++index) {
    const VkQueueFamilyProperties &properties = queue_families[index];
    if (properties.queueCount > 0 &&
        ((properties.queueFlags & queue_flags) == queue_flags)) {
      *valid_timestamp_bits = properties.timestampValidBits;
      return index;
    }
  }

  return absl::UnavailableError("cannot find queue family with required bits");
}

}  // namespace

void Driver::PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
  createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback;
}

bool Driver::CheckValidationLayerSupport(DynamicSymbols *symbols) {
  uint32_t layerCount;
  symbols->vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  symbols->vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
  
  const std::vector<const char*> validationLayers = {
      "VK_LAYER_KHRONOS_validation"
  };
  for (const char* layerName : validationLayers) {
      bool layerFound = false;

      for (const auto& layerProperties : availableLayers) {
          if (strcmp(layerName, layerProperties.layerName) == 0) {
              layerFound = true;
              break;
          }
      }

      if (!layerFound) {
          return false;
      }
  }

  return true;
}

absl::StatusOr<std::unique_ptr<Driver>> Driver::Create(
    const char *app_name, DynamicSymbols *symbols) {
  auto app_info = GetDefaultApplicationInfo(app_name);

  VkInstanceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.flags = 0;
  create_info.pApplicationInfo = &app_info;

#define DEBUG (1)
#if DEBUG
  const bool support_validation = CheckValidationLayerSupport(symbols);
  if (not support_validation) {
    fprintf(stderr, "validation layers requested, but not available!\n");
  } else {
    std::vector<const char*> applicationExtensions = {VK_EXT_DEBUG_REPORT_EXTENSION_NAME};
    create_info.enabledExtensionCount = static_cast<uint32_t>(applicationExtensions.size());
    create_info.ppEnabledExtensionNames = applicationExtensions.data();

    std::vector<const char*> validLayerNames = {
        "VK_LAYER_LUNARG_assistant_layer",
        "VK_LAYER_LUNARG_standard_validation",
        "VK_LAYER_KHRONOS_validation",
    };
    create_info.enabledLayerCount = static_cast<uint32_t>(validLayerNames.size());
    create_info.ppEnabledLayerNames = validLayerNames.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    PopulateDebugMessengerCreateInfo(debugCreateInfo);
    create_info.pNext =  (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
  }
#else
  create_info.enabledLayerCount = 0;
  // create_info.ppEnabledLayerNames = nullptr;
  create_info.enabledExtensionCount = 0;
  create_info.ppEnabledExtensionNames = nullptr;
  create_info.pNext = nullptr;
#endif

  VkInstance instance = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(symbols->vkCreateInstance(
      &create_info, /*pAllocator=*/nullptr, &instance));

  VkDebugUtilsMessengerEXT messenger;
#if DEBUG
  if (support_validation) {
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    PopulateDebugMessengerCreateInfo(debugCreateInfo);
    if (symbols->vkCreateDebugUtilsMessengerEXT(instance, &debugCreateInfo, /*pAllocator=*/nullptr, &messenger) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug messenger!");
    }
  }
#endif

  UVKC_RETURN_IF_ERROR(symbols->LoadFromInstance(instance));

  return absl::WrapUnique(new Driver(instance, *symbols, messenger));
}

Driver::~Driver() {
#if DEBUG
  symbols_.vkDestroyDebugUtilsMessengerEXT(instance_, messenger_, /*pAllocator=*/nullptr);
#endif
  symbols_.vkDestroyInstance(instance_, /*pAllocator=*/nullptr);
}

absl::StatusOr<std::vector<Driver::PhysicalDeviceInfo>>
Driver::EnumeratePhysicalDevices() {
  uint32_t count = 0;
  VK_RETURN_IF_ERROR(
      symbols_.vkEnumeratePhysicalDevices(instance_, &count, nullptr));

  std::vector<VkPhysicalDevice> devices(count);
  VK_RETURN_IF_ERROR(
      symbols_.vkEnumeratePhysicalDevices(instance_, &count, devices.data()));

  std::vector<PhysicalDeviceInfo> infos(count);
  for (int i = 0; i < count; ++i) {
    VkPhysicalDeviceSubgroupProperties subgroup_properties = {};
    subgroup_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup_properties.pNext = nullptr;

    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &subgroup_properties;

    symbols_.vkGetPhysicalDeviceProperties2(devices[i], &properties2);

    infos[i].handle = devices[i];
    infos[i].v10_properties = properties2.properties;
    infos[i].subgroup_properties = subgroup_properties;
  }

  return infos;
}

absl::StatusOr<std::unique_ptr<Device>> Driver::CreateDevice(
    const Driver::PhysicalDeviceInfo &physical_device,
    VkQueueFlags queue_flags) {
  uint32_t valid_timestamp_bits = 0;
  UVKC_ASSIGN_OR_RETURN(uint32_t queue_family_index,
                        SelectQueueFamily(physical_device.handle, queue_flags,
                                          &valid_timestamp_bits, symbols_));

  float queue_priority = 1.0;

  VkDeviceQueueCreateInfo queue_create_info = {};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.pNext = nullptr;
  queue_create_info.flags = 0;
  queue_create_info.queueFamilyIndex = queue_family_index;
  queue_create_info.queueCount = 1;
  queue_create_info.pQueuePriorities = &queue_priority;

  VkDeviceCreateInfo device_create_info = {};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.pNext = nullptr;
  device_create_info.flags = 0;
  device_create_info.queueCreateInfoCount = 1;
  device_create_info.pQueueCreateInfos = &queue_create_info;
  device_create_info.enabledLayerCount = 0;
  device_create_info.ppEnabledLayerNames = nullptr;
  device_create_info.enabledExtensionCount = 0;
  device_create_info.ppEnabledExtensionNames = nullptr;
  device_create_info.pEnabledFeatures = nullptr;

  VkDevice device;
  VK_RETURN_IF_ERROR(symbols_.vkCreateDevice(physical_device.handle,
                                             &device_create_info,
                                             /*pAllocator=*/nullptr, &device));
  return Device::Create(
      physical_device.handle, queue_family_index, valid_timestamp_bits,
      physical_device.v10_properties.limits.timestampPeriod, device, symbols_);
}

Driver::Driver(VkInstance instance, const DynamicSymbols &symbols, VkDebugUtilsMessengerEXT messenger)
    : instance_(instance), symbols_(symbols),  messenger_(messenger){}

}  // namespace vulkan
}  // namespace uvkc
