/*
This example should reproduce a possible bug in the Nvidia drivers,
It provokes a DeviceLost error with an endless shader loop and tries to create another device after that.
This fails with an InitializationFailed error on a GTX 1060 and just hangs with on a RTX 2060 Ti in vkCreateDevice,
so a real app isn't able to handle such errors.
*/

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.hpp>
#include <glfw/glfw3.h>
#include <iostream>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "vertex.vert.h"
#include "fragment.frag.h"

bool isInstanceExtensionAvailable(std::string const& ext)
{
	std::vector<std::string> avail_exts;
	for (auto const& e : vk::enumerateInstanceExtensionProperties())
		avail_exts.push_back(e.extensionName);
	return std::find(avail_exts.begin(), avail_exts.end(), ext) != avail_exts.end();
}

bool isInstanceLayerAvailable(std::string const& layer)
{
	std::vector<std::string> avail_layers;
	for (auto const& l : vk::enumerateInstanceLayerProperties())
		avail_layers.push_back(l.layerName);
	return std::find(avail_layers.begin(), avail_layers.end(), layer) != avail_layers.end();
}

template<typename T, std::size_t N>
std::vector<T> toVector(T const (&a)[N])
{
	return std::vector<T>(std::begin(a), std::end(a));
}

vk::UniqueShaderModule createShader(vk::Device dev, std::vector<std::uint32_t> const& spv)
{
	auto const shader_info{ vk::ShaderModuleCreateInfo{}
		.setCodeSize(spv.size() * sizeof(std::uint32_t))
		.setPCode(spv.data()) };
	return dev.createShaderModuleUnique(shader_info);
}

uint32_t selectMemoryTypeIndex(
	vk::PhysicalDevice phys_dev,
	vk::MemoryRequirements mem_req,
	vk::MemoryPropertyFlags preferred,
	vk::MemoryPropertyFlags required)
{
	auto const mem_props{ phys_dev.getMemoryProperties() };
	for (unsigned i = 0; i < VK_MAX_MEMORY_TYPES; ++i)
		if ((mem_req.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & preferred) == preferred)
			return i;
	if (required != preferred)
		for (unsigned i = 0; i < VK_MAX_MEMORY_TYPES; ++i)
			if ((mem_req.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & required) == required)
				return i;

	throw std::runtime_error{ "required memory type not available" };
}

class Scene
{
public:
	using Window = GLFWwindow;

	void initialize();
	void run();
	void shutdown();

private:
	void createWindowAndSurface();
	void initializeVKInstance();
	void selectQueueFamilyAndPhysicalDevice();
	void initializeDevice();

	void createSurface();
	void createSwapChainAndImages();
	void createSwapChainImageViews();

	void createPass();
	void createFramebuffer();
	void allocateCommandBuffers();
	void createShaderInterface();
	void createPipeline();
	void initSyncEntities();
	void buildCommandBuffer(uint32_t image_index);

	Window* m_window;
	uint32_t m_width = 1280;
	uint32_t m_height = 720;

	const vk::Format m_swapchain_format = vk::Format::eB8G8R8A8Unorm;
	const vk::Format m_depth_image_format = vk::Format::eD32Sfloat;
	const vk::PresentModeKHR m_present_mode = vk::PresentModeKHR::eFifo;
	const uint32_t m_sw_num_images = 2;

	vk::UniqueInstance m_instance;
	vk::PhysicalDevice m_phys_dev;
	uint32_t m_gq_fam_idx = -1;
	vk::UniqueDevice m_device;
	vk::Queue m_gr_queue;

	vk::UniqueSurfaceKHR m_surface;
	vk::UniqueSwapchainKHR m_swapchain;
	std::vector<vk::Image> m_swapchain_imgs;
	std::vector<vk::UniqueImageView> m_swapchain_img_views;

	vk::UniqueRenderPass m_render_pass;
	std::vector<vk::UniqueFramebuffer> m_framebuffers;
	vk::UniqueCommandPool m_cmd_b_pool;
	std::vector<vk::UniqueCommandBuffer> m_command_buffers;

	vk::UniqueShaderModule m_vert_shader;
	vk::UniqueShaderModule m_frag_shader;
	vk::UniquePipelineLayout m_pipeline_layout;
	vk::UniquePipeline m_pipeline;

	std::vector<vk::UniqueFence> m_fences;
	vk::UniqueSemaphore m_present_semaphore;
	vk::UniqueSemaphore m_draw_semaphore;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Scene::initialize()
{
	createWindowAndSurface();
	initializeVKInstance();
	selectQueueFamilyAndPhysicalDevice();
	initializeDevice();
	createSurface();
	createSwapChainAndImages();
	createSwapChainImageViews();

	createPass();
	createFramebuffer();
	allocateCommandBuffers();
	createShaderInterface();
	createPipeline();
	initSyncEntities();
}

void Scene::run()
{
	while (true)
	{
		glfwPollEvents();
		if (glfwWindowShouldClose(m_window))
			break;

		const uint32_t image_index = m_device->acquireNextImageKHR(*m_swapchain, UINT64_MAX, *m_draw_semaphore, {}).value;

		m_device->waitForFences(*m_fences[image_index], true, UINT64_MAX);
		m_device->resetFences(*m_fences[image_index]);

		buildCommandBuffer(image_index);

		const vk::PipelineStageFlags wait_mask = vk::PipelineStageFlagBits::eColorAttachmentOutput;

		vk::SubmitInfo submit_info{};
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &*m_command_buffers[image_index];
		submit_info.pSignalSemaphores = &*m_present_semaphore;
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitDstStageMask = &wait_mask;
		submit_info.pWaitSemaphores = &*m_draw_semaphore;
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = &*m_present_semaphore;

		m_gr_queue.submit(submit_info, *m_fences[image_index]);

		vk::PresentInfoKHR present_info{};
		present_info.pImageIndices = &image_index;
		present_info.pSwapchains = &*m_swapchain;
		present_info.pWaitSemaphores = &*m_present_semaphore;
		present_info.swapchainCount = 1;
		present_info.waitSemaphoreCount = 1;

		m_gr_queue.presentKHR(present_info);
	}
}

void Scene::shutdown()
{
	try
	{
		if (m_device)
			m_device->waitIdle();
	}
	catch (...)
	{}
	if (m_window)
		glfwDestroyWindow(m_window);
	glfwTerminate();
}

void glfwError(int ec, const char* emsg)
{
	std::cerr << "Error Code: " << ec << ", Error Msg: " << emsg << std::endl;
}

void Scene::createWindowAndSurface()
{
	glfwInit();
	glfwSetErrorCallback(glfwError);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	m_window = glfwCreateWindow(m_width,m_height,"",nullptr,nullptr);
	if (m_window == nullptr)
	{
		throw std::runtime_error("Window Creation failed!");
		glfwTerminate();
	}
}

void Scene::initializeVKInstance()
{
	std::vector<const char*> extensions;
	std::vector < const char*> layers;

	if (!isInstanceExtensionAvailable(VK_KHR_SURFACE_EXTENSION_NAME))
		throw std::runtime_error(std::string(VK_KHR_SURFACE_EXTENSION_NAME) + " is not available!");
	if (!isInstanceExtensionAvailable(VK_KHR_WIN32_SURFACE_EXTENSION_NAME))
		throw std::runtime_error(std::string(VK_KHR_SURFACE_EXTENSION_NAME) + " is not available!");
	extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
	extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);

//	if (isInstanceLayerAvailable("VK_LAYER_KHRONOS_validation"))
//		layers.push_back("VK_LAYER_KHRONOS_validation");

	vk::InstanceCreateInfo inst_ci{};

	inst_ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
	inst_ci.ppEnabledLayerNames = layers.data();
	inst_ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	inst_ci.ppEnabledExtensionNames = extensions.data();

	vk::ApplicationInfo app_info{};
	app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.pEngineName = "Test Engine";
	app_info.apiVersion = VK_MAKE_VERSION(1, 1, 0);

	inst_ci.pApplicationInfo = &app_info;
	m_instance = vk::createInstanceUnique(inst_ci);
}

void Scene::selectQueueFamilyAndPhysicalDevice()
{
	const uint32_t phys_idx = 0;
	const auto phys_devs = m_instance->enumeratePhysicalDevices();
	if (phys_devs.size() <= phys_idx)
		throw std::runtime_error("Invalid Physical Device Index provided!");
	m_phys_dev = phys_devs[phys_idx];

	const auto queue_fam_props = m_phys_dev.getQueueFamilyProperties();
	for (uint32_t i = 0; i < queue_fam_props.size(); ++i)
	{
		const auto prop = queue_fam_props[i];
		if (prop.queueFlags & vk::QueueFlagBits::eGraphics && prop.queueCount > 0)
		{
			m_gq_fam_idx = i;
			return;
		}
	}

	throw std::runtime_error("Can not find graphics family index!");
}

void Scene::initializeDevice()
{
	vk::DeviceCreateInfo dev_ci{};
	vk::DeviceQueueCreateInfo dev_q_ci{};
	float queue_prio = 1.0f;
	const std::vector<const char*> extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	dev_q_ci.queueCount = 1;
	dev_q_ci.pQueuePriorities = &queue_prio;
	dev_q_ci.queueFamilyIndex = m_gq_fam_idx;

	dev_ci.queueCreateInfoCount = 1;
	dev_ci.pQueueCreateInfos = &dev_q_ci;

	dev_ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	dev_ci.ppEnabledExtensionNames = extensions.data();

	m_device = m_phys_dev.createDeviceUnique(dev_ci);
	m_gr_queue = m_device->getQueue(m_gq_fam_idx, 0);
}

void Scene::createSurface()
{
	auto const create_info = vk::Win32SurfaceCreateInfoKHR{}
		.setHinstance(GetModuleHandle(nullptr))
		.setHwnd(glfwGetWin32Window(m_window));

	vk::UniqueSurfaceKHR surf_tmp = m_instance->createWin32SurfaceKHRUnique(create_info, nullptr);
	if (surf_tmp.get() == vk::SurfaceKHR{} || !m_phys_dev.getSurfaceSupportKHR(m_gq_fam_idx, *surf_tmp))
		throw std::runtime_error("Can not create Surface!");
	m_surface = std::move(surf_tmp);
}

void Scene::createSwapChainAndImages()
{
	auto const caps{ m_phys_dev.getSurfaceCapabilitiesKHR(*m_surface) };
	if (m_width != caps.currentExtent.width || m_height != caps.currentExtent.height)
		throw std::runtime_error{ "chosen image size not supported by window surface" };
	if (m_sw_num_images < caps.minImageCount)
		throw std::runtime_error{ "chosen image count is too small and not supported by the window surface" };
	if ((caps.maxImageCount != 0 && m_sw_num_images > caps.maxImageCount))
		throw std::runtime_error{ "chosen image count is too large and not supported by the window surface" };
	if (!(caps.supportedUsageFlags & vk::ImageUsageFlagBits::eColorAttachment))
		throw std::runtime_error{ "window surface cannot be used as color attachment" };

	bool format_found = false;
	for (auto const& surf_format : m_phys_dev.getSurfaceFormatsKHR(*m_surface))
	{
		if (surf_format.format == vk::Format::eUndefined || surf_format.format == m_swapchain_format)
		{
			format_found = true;
			break;
		}
	}
	if (!format_found)
		throw std::runtime_error{ "window surface not compatible with chosen color format" };

	bool present_mode_found = false;
	for (const auto& surf_mode : m_phys_dev.getSurfacePresentModesKHR(*m_surface))
		if (m_present_mode == surf_mode)
			present_mode_found = true;
	if (!present_mode_found)
		throw std::runtime_error("Chosen Present Mode is not supported!");

	vk::SwapchainCreateInfoKHR sw_ci{};
	sw_ci.setSurface(*m_surface);
	sw_ci.setMinImageCount(m_sw_num_images);
	sw_ci.setImageFormat(m_swapchain_format);
	sw_ci.setImageExtent(vk::Extent2D{ m_width, m_height });
	sw_ci.setImageArrayLayers(1);
	sw_ci.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
	sw_ci.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
	sw_ci.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
	sw_ci.setPresentMode(m_present_mode);
	sw_ci.setClipped(true);

	m_swapchain = m_device->createSwapchainKHRUnique(sw_ci);
	m_swapchain_imgs = m_device->getSwapchainImagesKHR(*m_swapchain);
}

void Scene::createSwapChainImageViews()
{
	vk::ImageSubresourceRange img_sb_range{};
	img_sb_range.aspectMask = vk::ImageAspectFlagBits::eColor;
	img_sb_range.levelCount = 1;
	img_sb_range.layerCount = 1;

	vk::ImageViewCreateInfo sw_imgv_ci{};
	sw_imgv_ci.subresourceRange = img_sb_range;
	sw_imgv_ci.format = m_swapchain_format;
	sw_imgv_ci.viewType = vk::ImageViewType::e2D;

	for (auto const sc_image : m_swapchain_imgs)
	{
		sw_imgv_ci.image = sc_image;
		m_swapchain_img_views.push_back(m_device->createImageViewUnique(sw_imgv_ci));
	}
}

void Scene::createPass()
{
	vk::AttachmentDescription color_att_desc{};
	color_att_desc.format      = m_swapchain_format;
	color_att_desc.samples     = vk::SampleCountFlagBits::e1;
	color_att_desc.loadOp      = vk::AttachmentLoadOp::eClear;
	color_att_desc.storeOp     = vk::AttachmentStoreOp::eStore;
	color_att_desc.finalLayout = vk::ImageLayout::ePresentSrcKHR;

	vk::AttachmentReference color_att_ref{};
	color_att_ref.attachment = 0;
	color_att_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::SubpassDescription subpass_desc{};
	subpass_desc.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subpass_desc.colorAttachmentCount = 1;
	subpass_desc.pColorAttachments = &color_att_ref;

	vk::RenderPassCreateInfo rp_ci{};
	rp_ci.attachmentCount = 1;
	rp_ci.pAttachments = &color_att_desc;
	rp_ci.subpassCount = 1;
	rp_ci.pSubpasses = &subpass_desc;

	m_render_pass = m_device->createRenderPassUnique(rp_ci);
}


void Scene::createFramebuffer()
{
	const vk::ImageView att_0 = nullptr;

	vk::FramebufferCreateInfo fb_ci{};
	fb_ci.renderPass = *m_render_pass;
	fb_ci.attachmentCount = 1;
	fb_ci.width = m_width;
	fb_ci.height = m_height;
	fb_ci.layers = 1;

	for (uint32_t i = 0;i < m_sw_num_images;++i)
	{
		fb_ci.pAttachments = &*m_swapchain_img_views[i];
		m_framebuffers.push_back(m_device->createFramebufferUnique(fb_ci));
	}
}

void Scene::allocateCommandBuffers()
{
	vk::CommandPoolCreateInfo cmd_pool_ci{};
	cmd_pool_ci.queueFamilyIndex = m_gq_fam_idx;
	cmd_pool_ci.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

	m_cmd_b_pool = m_device->createCommandPoolUnique(cmd_pool_ci);

	vk::CommandBufferAllocateInfo cmd_b_ai{};
	cmd_b_ai.commandBufferCount = m_sw_num_images;
	cmd_b_ai.commandPool = *m_cmd_b_pool;
	cmd_b_ai.level = vk::CommandBufferLevel::ePrimary;

	m_command_buffers = m_device->allocateCommandBuffersUnique(cmd_b_ai);
}


void Scene::createShaderInterface()
{
	vk::PipelineLayoutCreateInfo pl_ci{};
	m_pipeline_layout = m_device->createPipelineLayoutUnique(pl_ci);
}

void Scene::createPipeline()
{
	vk::PipelineVertexInputStateCreateInfo vt_inp_ci{};

	vk::PipelineColorBlendAttachmentState cbas_ci{};
	cbas_ci.colorWriteMask = static_cast<vk::ColorComponentFlags>(0xf);

	vk::PipelineColorBlendStateCreateInfo cbs_ci{};
	cbs_ci.attachmentCount = 1;
	cbs_ci.pAttachments = &cbas_ci;

	vk::PipelineDepthStencilStateCreateInfo dss_ci{};

	vk::PipelineInputAssemblyStateCreateInfo as_ci{};
	as_ci.topology = vk::PrimitiveTopology::eTriangleList;

	vk::PipelineMultisampleStateCreateInfo mss_ci{};
	mss_ci.rasterizationSamples = vk::SampleCountFlagBits::e1;

	vk::PipelineRasterizationStateCreateInfo rss_ci{};
	rss_ci.cullMode = vk::CullModeFlagBits::eNone;
	rss_ci.polygonMode = vk::PolygonMode::eFill;
	rss_ci.lineWidth = 1.0f;

	m_vert_shader = createShader(*m_device, toVector(::Vertex_vert));
	m_frag_shader = createShader(*m_device, toVector(::Fragment_frag));

	std::vector <vk::PipelineShaderStageCreateInfo> sh_stages;
	vk::PipelineShaderStageCreateInfo ss_ci{};
	ss_ci.pName = "main";

	ss_ci.module = *m_vert_shader;
	ss_ci.stage = vk::ShaderStageFlagBits::eVertex;
	sh_stages.push_back(ss_ci);

	ss_ci.module = *m_frag_shader;
	ss_ci.stage = vk::ShaderStageFlagBits::eFragment;
	sh_stages.push_back(ss_ci);

	vk::PipelineViewportStateCreateInfo vps_ci{};
	vk::Viewport viewport{};
	viewport.width = static_cast<float>(m_width);
	viewport.height = static_cast<float>(m_height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vps_ci.viewportCount = 1;
	vps_ci.pViewports = &viewport;

	vk::Rect2D scissor{};
	scissor.extent.width = m_width;
	scissor.extent.height = m_height;
	vps_ci.scissorCount = 1;
	vps_ci.pScissors = &scissor;

	vk::GraphicsPipelineCreateInfo gp_ci{};
	gp_ci.pVertexInputState = &vt_inp_ci;
	gp_ci.layout = *m_pipeline_layout;
	gp_ci.pColorBlendState = &cbs_ci;
	gp_ci.pDepthStencilState = &dss_ci;
	gp_ci.pInputAssemblyState = &as_ci;
	gp_ci.pMultisampleState = &mss_ci;
	gp_ci.pRasterizationState = &rss_ci;
	gp_ci.stageCount = static_cast<uint32_t>(sh_stages.size());
	gp_ci.pStages = sh_stages.data();
	gp_ci.renderPass = *m_render_pass;
	gp_ci.subpass = 0;
	gp_ci.pViewportState = &vps_ci;

	m_pipeline = m_device->createGraphicsPipelineUnique({}, gp_ci).value;
}

void Scene::initSyncEntities()
{
	vk::FenceCreateInfo f_ci{};
	f_ci.flags = vk::FenceCreateFlagBits::eSignaled;

	for (uint32_t i = 0;i < m_sw_num_images;++i)
		m_fences.push_back(m_device->createFenceUnique(f_ci));

	m_draw_semaphore = m_device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
	m_present_semaphore = m_device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
}

void Scene::buildCommandBuffer(uint32_t image_index)
{
	vk::CommandBufferBeginInfo cmd_begin_info{};
	auto& cmd = *m_command_buffers[image_index];

	cmd.begin(cmd_begin_info);
	const std::array<float,4> clear_color{0, 0, 1, 1};

	const vk::ClearValue clear_value = { vk::ClearColorValue(clear_color) };

	vk::RenderPassBeginInfo rp_begin_info{};
	rp_begin_info.framebuffer = *m_framebuffers[image_index];
	rp_begin_info.renderArea = vk::Rect2D({ 0,0 }, { m_width,m_height });
	rp_begin_info.renderPass = *m_render_pass;
	rp_begin_info.clearValueCount = 1;
	rp_begin_info.pClearValues = &clear_value;

	cmd.beginRenderPass(rp_begin_info, vk::SubpassContents::eInline);
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_pipeline);
	cmd.draw(3, 1, 0, 0);
	cmd.endRenderPass();

	cmd.end();
}

int main()
{
	SetProcessDPIAware();

	// provoke DeviceLost
	int return_value = 0;
	bool device_lost = false;
	{
		Scene scene;
		try
		{
			scene.initialize();
			scene.run();
		}
		catch (vk::DeviceLostError const&)
		{
			std::cerr << "Device Lost, re-init..." << std::endl;
			device_lost = true;
		}
		catch (std::exception& e)
		{
			std::cerr << "Error Occurred: " << e.what() << std::endl;
			return_value = 1;
		}
		scene.shutdown();
		if (!device_lost)
			return return_value;
	}

	// re-init
	Scene scene;
	try
	{
		scene.initialize();
		std::cout << "re-initialization successful" << std::endl;
	}
	catch (std::exception& e)
	{
		std::cerr << "Error Occurred: " << e.what() << std::endl;
		return_value = 1;
	}
	scene.shutdown();
	std::system("pause");
	return return_value;
}
