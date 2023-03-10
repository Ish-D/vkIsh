#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>

#include <algorithm>
#include <set>
#include <cstring>

#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.hpp>

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}


int main()	
{
    constexpr uint32_t width = 800;
    constexpr uint32_t height = 600;

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(width, height, "vkIsh", nullptr, nullptr);
    glfwSetKeyCallback(window, key_callback);

    vk::ApplicationInfo appInfo("vkIsh", VK_MAKE_VERSION(1, 0, 0), "N/A", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_3);

    auto glfwExtensionCount = 0u;
    auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> glfwExtensionsVector(glfwExtensions, glfwExtensions + glfwExtensionCount);
    glfwExtensionsVector.push_back("VK_EXT_debug_utils");
    std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };

    auto instance = vk::createInstanceUnique(vk::InstanceCreateInfo{ {}, &appInfo, validationLayers, glfwExtensionsVector });
    auto dispatch_loader = vk::DispatchLoaderDynamic(*instance, vkGetInstanceProcAddr);

    auto messenger = instance->createDebugUtilsMessengerEXTUnique(
        vk::DebugUtilsMessengerCreateInfoEXT{ {},
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            debugCallback },
        nullptr, dispatch_loader);

    VkSurfaceKHR surfaceTmp;
    VkResult err = glfwCreateWindowSurface(*instance, window, nullptr, &surfaceTmp);

    vk::UniqueSurfaceKHR surface(surfaceTmp, *instance);

    const auto physicalDevices = instance->enumeratePhysicalDevices();

    for (const auto& device : physicalDevices)
    {
        std::cout << device.getProperties().deviceName << "\n";
    }

    auto physicalDevice = physicalDevices[std::distance(physicalDevices.begin(), std::find_if(physicalDevices.begin(), physicalDevices.end(),
        [](const vk::PhysicalDevice& physicalDevice) {return strstr(physicalDevice.getProperties().deviceName, "NVIDIA"); }))];

    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    size_t graphicsQueueFamilyIndex = std::distance(queueFamilyProperties.begin(),std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(), 
        [](vk::QueueFamilyProperties const& qfp) {return qfp.queueFlags & vk::QueueFlagBits::eGraphics;}));

    size_t presentQueueFamilyIndex = 0u;
    for (auto i = 0ul; i<queueFamilyProperties.size(); i++)
    {
	    if (physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), surface.get()))
	    {
            presentQueueFamilyIndex = i;
	    }
    }

    std::set<uint32_t> uniqueQueueFamilyIndices = { static_cast<uint32_t>(graphicsQueueFamilyIndex), static_cast<uint32_t>(presentQueueFamilyIndex) };
    std::vector<uint32_t> familyIndices = { uniqueQueueFamilyIndices.begin(), uniqueQueueFamilyIndices.end() };

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    const float queuePriority = 0.0f;
    for (auto& queueFamilyIndex : uniqueQueueFamilyIndices)
    {
        queueCreateInfos.push_back(vk::DeviceQueueCreateInfo{ vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(queueFamilyIndex), 1, &queuePriority });
    }

    const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    vk::UniqueDevice device = physicalDevice.createDeviceUnique(
        vk::DeviceCreateInfo(vk::DeviceCreateFlags(), static_cast<uint32_t>(queueCreateInfos.size()), queueCreateInfos.data(), 
            0u, nullptr, static_cast<uint32_t>(deviceExtensions.size()), deviceExtensions.data()));

    uint32_t inFlightFrames = 2;
    struct SM
    {
        vk::SharingMode sharingMode;
        uint32_t familyIndicesCount;
        uint32_t* pFamilyIndices;
    }
    sharingModeUtil{ (graphicsQueueFamilyIndex != presentQueueFamilyIndex) ?
        SM{vk::SharingMode::eConcurrent, 2u, familyIndices.data()} :
        SM{vk::SharingMode::eExclusive, 0u, static_cast<uint32_t*>(nullptr)} };

    auto capabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    auto formats = physicalDevice.getSurfaceFormatsKHR(*surface);

    auto format = vk::Format::eB8G8R8A8Unorm;
    auto extent = vk::Extent2D{ width, height };

    vk::SwapchainCreateInfoKHR swapChainCreateInfo({}, surface.get(), inFlightFrames, format, vk::ColorSpaceKHR::eSrgbNonlinear, extent,
        1, vk::ImageUsageFlagBits::eColorAttachment, sharingModeUtil.sharingMode, sharingModeUtil.familyIndicesCount, sharingModeUtil.pFamilyIndices,
        vk::SurfaceTransformFlagBitsKHR::eIdentity, vk::CompositeAlphaFlagBitsKHR::eOpaque, vk::PresentModeKHR::eFifo, true, nullptr);

    auto swapChain = device->createSwapchainKHRUnique(swapChainCreateInfo);
    std::vector<vk::Image> swapChainImages = device->getSwapchainImagesKHR(swapChain.get());

    std::vector<vk::UniqueImageView> imageViews;
    imageViews.reserve(swapChainImages.size());
    for (auto image : swapChainImages)
    {
        vk::ImageViewCreateInfo imageViewCreateInfo(vk::ImageViewCreateFlags(), image, vk::ImageViewType::e2D, format,
            vk::ComponentMapping{ vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eA },
            vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        imageViews.push_back(device->createImageViewUnique(imageViewCreateInfo));
    }

    std::string kShaderSource = R"vertexshader(
        #version 450
        #extension GL_ARB_separate_shader_objects : enable
        out gl_PerVertex {
            vec4 gl_Position;
        };
        layout(location = 0) out vec3 fragColor;
        vec2 positions[3] = vec2[](
            vec2(0.0, -0.5),
            vec2(0.5, 0.5),
            vec2(-0.5, 0.5)
        );
        vec3 colors[3] = vec3[](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );
        void main() {
            gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
            fragColor = colors[gl_VertexIndex];
        }
        )vertexshader";

    std::string fragmentShader = R"fragmentShader(
        #version 450
        #extension GL_ARB_separate_shader_objects : enable
        layout(location = 0) in vec3 fragColor;
        layout(location = 0) out vec4 outColor;
        void main() {
            outColor = vec4(fragColor, 1.0);
        }
        )fragmentShader";

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    shaderc::SpvCompilationResult vertShaderModule = compiler.CompileGlslToSpv(kShaderSource, shaderc_glsl_vertex_shader, "vertex shader", options);
    if (vertShaderModule.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << vertShaderModule.GetErrorMessage();
    }
    auto vertShaderCode = std::vector<uint32_t>{ vertShaderModule.cbegin(), vertShaderModule.cend() };
    auto vertSize = std::distance(vertShaderCode.begin(), vertShaderCode.end());
    auto vertShaderCreateInfo =
        vk::ShaderModuleCreateInfo{ {}, vertSize * sizeof(uint32_t), vertShaderCode.data() };
    auto vertexShaderModule = device->createShaderModuleUnique(vertShaderCreateInfo);

    shaderc::SpvCompilationResult fragShaderModule = compiler.CompileGlslToSpv( fragmentShader, shaderc_glsl_fragment_shader, "fragment shader", options);
    if (fragShaderModule.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << fragShaderModule.GetErrorMessage();
    }
    auto fragShaderCode = std::vector<uint32_t>{ fragShaderModule.cbegin(), fragShaderModule.cend() };
    auto fragSize = std::distance(fragShaderCode.begin(), fragShaderCode.end());
    auto fragShaderCreateInfo =
        vk::ShaderModuleCreateInfo{ {}, fragSize * sizeof(uint32_t), fragShaderCode.data() };
    auto fragmentShaderModule = device->createShaderModuleUnique(fragShaderCreateInfo);

    auto vertShaderStageInfo = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eVertex, *vertexShaderModule, "main" };
    auto fragShaderStageInfo = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eFragment, *fragmentShaderModule, "main" };
    auto pipelineShaderStages = std::vector<vk::PipelineShaderStageCreateInfo>{ vertShaderStageInfo, fragShaderStageInfo };

    auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo{ {}, 0u, nullptr, 0u, nullptr };
    auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo{ {}, vk::PrimitiveTopology::eTriangleList, false };

    auto viewport = vk::Viewport{ 0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f };
    auto scissor = vk::Rect2D{ {0,0}, extent };
    auto viewportState = vk::PipelineViewportStateCreateInfo{ {}, 1, &viewport, 1, &scissor };
    auto rasterizer = vk::PipelineRasterizationStateCreateInfo{ {}, false, false, vk::PolygonMode::eFill, {}, vk::FrontFace::eClockwise,
		{}, {}, {}, {}, 1.0f };
    auto multisampling = vk::PipelineMultisampleStateCreateInfo{ {}, vk::SampleCountFlagBits::e1, false, 1.0 };
    auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState{
    	{}, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
            vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
    auto colorblending = vk::PipelineColorBlendStateCreateInfo{ {}, false, vk::LogicOp::eCopy,   1, &colorBlendAttachment };
    auto pipelineLayout = device->createPipelineLayoutUnique({}, nullptr);
    auto colorAttachment = vk::AttachmentDescription{ {}, format, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eStore, {}, {}, {}, vk::ImageLayout::ePresentSrcKHR };
    auto colorAttachmentRef = vk::AttachmentReference{ 0, vk::ImageLayout::eColorAttachmentOptimal };
    auto subpass = vk::SubpassDescription{ {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorAttachmentRef };

    auto semaphoreCreateInfo = vk::SemaphoreCreateInfo{};
    auto imageAvailableSemaphore = device->createSemaphoreUnique(semaphoreCreateInfo);
    auto renderFinishedSemaphore = device->createSemaphoreUnique(semaphoreCreateInfo);

    auto subpassDependency = vk::SubpassDependency{ VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite };
    auto renderpass = device->createRenderPassUnique(vk::RenderPassCreateInfo{ {}, 1, &colorAttachment, 1, &subpass, 1, &subpassDependency });

    auto pipelineCreateInfo = vk::GraphicsPipelineCreateInfo{ {}, 2, pipelineShaderStages.data(), &vertexInputInfo, &inputAssembly, nullptr, &viewportState,
        &rasterizer, &multisampling, nullptr, &colorblending, nullptr, *pipelineLayout, *renderpass, 0 };
    auto pipeline = device->createGraphicsPipelineUnique({}, pipelineCreateInfo).value;

    auto framebuffers = std::vector<vk::UniqueFramebuffer>(inFlightFrames);
    for(size_t i = 0; i<imageViews.size(); i++)
    {
        framebuffers[i] = device->createFramebufferUnique(vk::FramebufferCreateInfo{ {}, *renderpass, 1, &(*imageViews[i]), extent.width, extent.height, 1 });
    }

    auto commandPoolUnique = device->createCommandPoolUnique({ {}, static_cast<uint32_t>(graphicsQueueFamilyIndex) });
    std::vector<vk::UniqueCommandBuffer> commandBuffers = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(commandPoolUnique.get(),
        vk::CommandBufferLevel::ePrimary, static_cast<uint32_t>(framebuffers.size())));

    auto deviceQueue = device->getQueue(static_cast<uint32_t>(graphicsQueueFamilyIndex), 0);
    auto presentQueue = device->getQueue(static_cast<uint32_t>(presentQueueFamilyIndex), 0);

    for (size_t i = 0; i<commandBuffers.size(); i++)
    {
        auto beginInfo = vk::CommandBufferBeginInfo{};
        commandBuffers[i]->begin(beginInfo);
        vk::ClearValue clearValues{};
        auto renderpassBeginInfo = vk::RenderPassBeginInfo{ renderpass.get(), framebuffers[i].get(), vk::Rect2D{{0,0}, extent}, 1, &clearValues };

        commandBuffers[i]->beginRenderPass(renderpassBeginInfo, vk::SubpassContents::eInline);
        commandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
        commandBuffers[i]->draw(3, 1, 0, 0);
        commandBuffers[i]->endRenderPass();
        commandBuffers[i]->end();
    }

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        auto imageIndex = device->acquireNextImageKHR(swapChain.get(),
            std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore.get(), {});

        vk::PipelineStageFlags waitStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;

        auto submitInfo = vk::SubmitInfo{ 1, &imageAvailableSemaphore.get(), &waitStageMask, 1,
            &commandBuffers[imageIndex.value].get(), 1, &renderFinishedSemaphore.get() };

        deviceQueue.submit(submitInfo, {});

        auto presentInfo = vk::PresentInfoKHR{ 1, &renderFinishedSemaphore.get(), 1,
            &swapChain.get(), &imageIndex.value };
        auto result = presentQueue.presentKHR(presentInfo);

        device->waitIdle();
    }
}