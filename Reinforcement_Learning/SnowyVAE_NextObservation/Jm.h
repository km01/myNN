#pragma once
/* https://www.inflearn.com/course/c-2 */
#include <cassert>
#include <iostream>
#include <chrono>
#include <cassert>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <math.h>
#include <vector>
#include <thread>         // std::this_thread::sleep_for
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include <map>
#include <iomanip>

class Timer
{
private:
	// Type aliases to make accessing nested type easier
	using clock_t = std::chrono::high_resolution_clock;
	//using second_t = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<clock_t> start_time, end_time;

public:
	Timer()
		: start_time(clock_t::now())
	{}

	void reset() {
		start_time = clock_t::now();
	}
	void start() {
		reset();
	}
	void stop() {
		end_time = clock_t::now();
	}
	double getElapsedMilli() const {
		const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

		return static_cast<double>(elapsed);
	}
	double stopAndGetElapsedMilli() {
		stop();

		return getElapsedMilli();
	}
};


template<typename T>
class Vector2
{
public:
	union {
		struct { T x, y; };
		struct { T v0, v1; };
		T data[2];
	};

	Vector2() {}
	Vector2(const T& x_input, const T& y_input) : x(x_input), y(y_input) {}
	~Vector2() {}

	void operator += (const Vector2<T>& v) {
		x += v.x;
		y += v.y;
	}

	void operator -= (const Vector2<T>& v) {
		x -= v.x;
		y -= v.y;
	}

	void operator *= (const T& s) {
		x *= s;
		y *= s;
	}

	void operator /= (const T& s) {
		const T one_over_s = T(1) / s;
		x *= one_over_s;
		y *= one_over_s;
	}

	Vector2<T> operator + (const Vector2<T>& v) {
		return Vector2<T>(x + v.x, y + v.y);
	}

	Vector2<T> operator - (const Vector2<T>& v) {
		return Vector2<T>(x - v.x, y - v.y);
	}

	Vector2<T> operator * (const T& a) {
		return Vector2<T>(x * a, y * a);
	}

	Vector2<T> operator / (const T& a) {
		const T one_over_a = T(1) / a;
		return Vector2<T>(x * one_over_a, y * one_over_a);
	}

	Vector2<T> operator - () {
		return Vector2<T>(-x, -y);
	}

	float& operator [] (const int& ix) {
		assert(ix >= 0);
		assert(ix < 2);

		return data[ix];
	}

	const float& operator [] (const int& ix) const {
		assert(ix >= 0);
		assert(ix < 2);

		return data[ix];
	}

	friend std::ostream& operator << (std::ostream& out, const Vector2<T>& vec) {
		out << vec.x << " " << vec.y;
		return out;
	}
};

template<typename T>
class Vector3 {
public:
	union {
		struct { T x, y, z; };
		struct { T v0, v1, v2; };
		struct { T r, g, b; };
		T data[3];
		T rgb[3];
	};
	Vector3() {}

	Vector3(const T& _x, const T& _y, const T& _z) : x(_x), y(_y), z(_z) {}

	~Vector3() {}

	void operator += (const Vector3<T>& v) {
		x += v.x;
		y += v.y;
		z += v.z;
	}

	void operator -= (const Vector3<T>& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
	}

	void operator *= (const T& s) {
		x *= s;
		y *= s;
	}

	void operator /= (const T& s)
	{
		const T one_over_s = T(1) / s;
		x *= one_over_s;
		y *= one_over_s;
	}

	Vector3<T> operator + (const Vector3<T>& v) const {
		return Vector3<T>(x + v.x, y + v.y, z + v.z);
	}

	Vector3<T> operator - (const Vector3<T>& v) const {
		return Vector3<T>(x - v.x, y - v.y, z - v.z);
	}

	Vector3<T> operator * (const T& a) const {
		return Vector3<T>(x * a, y * a, z * a);
	}

	Vector3<T> operator / (const T& a) const {
		const T one_over_a = T(1) / a;
		return Vector3<T>(x * one_over_a, y * one_over_a, z * one_over_a);
	}

	T& operator [] (const int& ix) {
		assert(ix >= 0);
		assert(ix < 3);

		return data[ix];
	}

	const T& operator [] (const int& ix) const {
		assert(ix >= 0);
		assert(ix < 3);

		return data[ix];
	}
};

class rgb : public Vector3<double> {
	using BASE = Vector3<double>;

public:

	rgb(const double& fr, const double& fg, const double& fb) : BASE(fr, fg, fb) {}

	// integer type rgb values are divided by 255.0f because OpenGL uses real rgb values.
	rgb(const int& cr, const int& cg, const int& cb) : BASE(static_cast<double>(cr) / 255.0, static_cast<double>(cg) / 255.0, static_cast<double>(cb) / 255.0) {}
	rgb(const rgb& _rgb) : BASE(_rgb.r, _rgb.g, _rgb.b) {}
	~rgb() {}
};
namespace Colors
{
	// RGB color table http://www.rapidtables.com/web/color/RGB_Color.htm
	const rgb red(255, 0, 0);
	const rgb green(0, 255, 0);
	const rgb blue(0, 0, 255);
	const rgb skyblue(178, 255, 255);
	const rgb gray(128, 128, 128);
	const rgb yellow(255, 255, 0);
	const rgb olive(128, 128, 0);
	const rgb black(0, 0, 0);
	const rgb white(255, 255, 255);
	const rgb gold(255, 223, 0);
	const rgb silver(192, 192, 192);
}

using vec2 = Vector2<double>;
using vec3 = Vector3<double>;

class Game
{
public:
	int width = 640;
	int height = 480;
	GLFWwindow* glfw_window = nullptr;
	Timer timer;

	double spf = 1.0 / 60.0;		 // second(s) per frame

	// control options
	std::map<int, bool> key_status;  // key_id, is_pressed
	std::map<int, bool> mbtn_status; // mouse_button_id, is_pressed
	bool draw_grid = false;
	
	Game()
	{}

	Game(const std::string& _title, const int& _width, const int& _height,
		const bool& use_full_screen = false, const int& display_ix = 0) {
		init(_title, _width, _height, use_full_screen, display_ix);
	}

	~Game() {
		glfwDestroyWindow(glfw_window); // cannot 'delete' glfw_window
	}

	Game& init(const std::string& _title, const int& _width, const int& _height,
		const bool& use_full_screen = false, const int& display_ix = 0) {
		if (glfw_window != nullptr) {
			std::cout << "Skip second initialization" << std::endl;
			return *this;
		}

		if (!glfwInit()) reportErrorAndExit(__FUNCTION__, "glfw initialization");

		const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

		int num_monitors;
		GLFWmonitor** monitors = glfwGetMonitors(&num_monitors);

		if (!use_full_screen) {
			glfw_window = glfwCreateWindow(_width, _height, _title.c_str(), NULL, NULL);

			// center window
			int display_w, display_h;
			glfwMakeContextCurrent(glfw_window);
			glfwGetFramebufferSize(glfw_window, &display_w, &display_h);

			width = display_w;
			height = display_h;

			//if (num_monitors == 3) // find center display
			//	glfwSetWindowPos(glfw_window, (mode->width - display_w) / 2 - mode->width, (mode->height - display_h) / 2);
			//else
			//	glfwSetWindowPos(glfw_window, (mode->width - display_w) / 2, (mode->height - display_h) / 2);
			glfwSetWindowPos(glfw_window, (mode->width - display_w) / 2, (mode->height - display_h) / 2);
		}
		else {
			if (display_ix < num_monitors) // display_ix is valid
				glfw_window = glfwCreateWindow(mode->width, mode->height, _title.c_str(), monitors[display_ix], NULL);
			else
				glfw_window = glfwCreateWindow(mode->width, mode->height, _title.c_str(), glfwGetPrimaryMonitor(), NULL);

			// full screen resolution
			width = mode->width;
			height = mode->height;
		}

		if (!glfw_window) reportErrorAndExit(__FUNCTION__, "Window initialization");

		glfwMakeContextCurrent(glfw_window);

		// Initialize GLEW
		glewExperimental = true; // Needed for core profile
		if (glewInit() != GLEW_OK) reportErrorAndExit(__FUNCTION__, "glew initialization");

		const float aspect_ratio = (float)width / (float)height;
		glViewport(0, 0, width, height);
		glOrtho(-aspect_ratio, aspect_ratio, -1.0, 1.0, -1.0, 1.0);
		std::cout << "Display width = " << width << " height = " << height <<
			" Aspect ratio is " << aspect_ratio << std::endl;

		glfwWindowHint(GLFW_SAMPLES, 32);
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
		glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		return *this; // chaining (See TBC++ 8.7)
	}

	void reportErrorAndExit(const std::string& function_name, const std::string& message) {
		std::cout << "Error: " << function_name << " " << message << std::endl;

		glfwTerminate();
		getchar(); // pause to read error message
		exit(1);
	}

	bool isKeyPressed(const int& key) {
		if (key_status.count(key) <= 0) key_status[key] = false;

		if (glfwGetKey(glfw_window, key) == GLFW_PRESS)
			key_status[key] = true;
		else
			key_status[key] = false;

		return key_status[key];
	}
	bool isKeyReleased(const int& key) {
		if (key_status.count(key) <= 0) key_status[key] = false;

		if (glfwGetKey(glfw_window, key) == GLFW_RELEASE)
			key_status[key] = false;
		else
			key_status[key] = true;

		return key_status[key];
	}
	bool isKeyPressedAndReleased(const int& key) {
		if (key_status.count(key) <= 0) key_status[key] = false; // register key to map

		if (glfwGetKey(glfw_window, key) == GLFW_RELEASE)
		{
			if (key_status[key] == true) {
				key_status[key] = false;
				return true;
			}
			else {
				key_status[key] = false;
				return false;
			}
		}
		else {
			key_status[key] = true;
			return false;
		}
	}

	bool isMouseButtonPressed(const int& key) {
		if (mbtn_status.count(key) <= 0) mbtn_status[key] = false;

		if (glfwGetMouseButton(glfw_window, key) == GLFW_PRESS)
			mbtn_status[key] = true;
		else
			mbtn_status[key] = false;

		return mbtn_status[key];
	}
	bool isMouseButtonReleased(const int& key) {
		if (mbtn_status.count(key) <= 0) mbtn_status[key] = false;

		if (glfwGetMouseButton(glfw_window, key) == GLFW_RELEASE)
			mbtn_status[key] = false;
		else
			mbtn_status[key] = true;

		return mbtn_status[key];
	}
	bool isMouseButtonPressedAndReleased(const int& mbtn) {
		if (mbtn_status.count(mbtn) <= 0) mbtn_status[mbtn] = false; // register key to map

		if (glfwGetMouseButton(glfw_window, mbtn) == GLFW_RELEASE)
		{
			if (mbtn_status[mbtn] == true) {
				mbtn_status[mbtn] = false;
				return true;
			}
			else {
				mbtn_status[mbtn] = false;
				return false;
			}
		}
		else {
			mbtn_status[mbtn] = true;
			return false;
		}
	}

	vec2 getCursorPos(const bool& screen_coordinates = true) {
		double x_pos, y_pos;
		glfwGetCursorPos(glfw_window, &x_pos, &y_pos);
		// Note that (0, 0) is left up corner. 
		// This is different from our screen coordinates.
		// 0 <= x <= width - 1
		// height - 1 >= y >= 0 

		if (screen_coordinates) // assumes width >= height
		{
			// upside down y direction
			y_pos = height - y_pos - 1; // 0 <= y <= height - 1

			// rescale and translate zero to center
			y_pos = y_pos / (height - 1); //  0.0 <= y <= 1.0
			y_pos = y_pos * 2.0;		  //  0.0 <= y <= 2.0
			y_pos = y_pos - 1.0;		  // -1.0 <= y <= 1.0

			x_pos = (x_pos / (width - 1) * 2.0 - 1.0) * width / height;

			return vec2(static_cast<float>(x_pos), static_cast<float>(y_pos));
		}
		else
		{
			return vec2(static_cast<float>(x_pos), static_cast<float>(y_pos));
		}
	}

	double getTimeStep() {
		return spf;
	}

	virtual bool isEnd() = 0;
	void run() {

		if (glfw_window == nullptr)
			init("This is my digital canvas!", 500, 500, false); // initialize with default setting
		while (!glfwWindowShouldClose(glfw_window))// main loop
		{
			if (isKeyPressed(GLFW_KEY_ESCAPE)) {
				std::this_thread::sleep_for(std::chrono::milliseconds(200));

				std::cout << "ESC key ends main loop" << std::endl;
				break;
			}

			timer.start();

			// pre draw
			glfwMakeContextCurrent(glfw_window);
			glClearColor(1, 1, 1, 1);			 // while background
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glMatrixMode(GL_MODELVIEW);

			glPushMatrix();

			update();

			glPopMatrix();

			// post draw
			glfwSwapBuffers(glfw_window); // double buffering
			//glfwSetInputMode(glfw_window, GLFW_STICKY_KEYS, GLFW_FALSE); // not working 
			glfwPollEvents();

			const double dt = timer.stopAndGetElapsedMilli();

			//Debugging
			//std::cout << dt << std::endl;

			if (dt < spf) // to prevent too high fps
			{
				const auto time_to_sleep = static_cast<int>((spf - dt) * 1000.0f);
				std::this_thread::sleep_for(std::chrono::milliseconds(time_to_sleep));

				//Debugging
				std::cout << "sleep " << time_to_sleep << std::endl;
			}
			if (isEnd()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
				timer.start();
				glfwMakeContextCurrent(glfw_window);
				glClearColor(1, 1, 1, 1);			 // while background
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				glMatrixMode(GL_MODELVIEW);
				glPushMatrix();
				update();
				glPopMatrix();
				glfwSwapBuffers(glfw_window);
				glfwPollEvents();
				break;
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		glfwTerminate();
	}
	virtual void update() {}
	void drawBitmapText(char* str, float x, float y, float z)
	{
		glColor3dv(Colors::black.data);
		glRasterPos3f(x, y, z);

		while (*str)
		{
			glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_10, *str);

			str++;
		}
	}
};